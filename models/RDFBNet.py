import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .pvtv2_encoder import pvt_v2_b1, pvt_v2_b2
except ImportError:
    from pvtv2_encoder import pvt_v2_b1, pvt_v2_b2


def conv3x3(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
    )


def conv_bn_relu(in_channels, out_channels, stride=1):
    return nn.Sequential(
        conv3x3(in_channels, out_channels, stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class EfficientChannelAttention(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x):
        weights = self.global_pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.channel_conv(weights).transpose(-1, -2).unsqueeze(-1)
        return x * torch.sigmoid(weights)


class StripContextBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.horizontal_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.vertical_pool = nn.AdaptiveAvgPool2d((1, None))
        self.horizontal_projection = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.vertical_projection = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU()

    def forward(self, x):
        _, _, height, width = x.shape
        horizontal = self.horizontal_projection(self.horizontal_pool(x))
        vertical = self.vertical_projection(self.vertical_pool(x))
        context = horizontal.expand(-1, -1, height, width)
        context = context + vertical.expand(-1, -1, height, width)
        return self.activation(self.norm(context))


class CFER(nn.Module):
    """Contextual Feature Enhancement and Recalibration."""

    def __init__(self, channels):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
        )
        branch_channels = channels // 2
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, branch_channels, 1),
            nn.BatchNorm2d(branch_channels),
            nn.PReLU(),
        )
        self.dilated_branch2 = self._dilated_branch(channels, branch_channels, dilation=2)
        self.dilation2_query = nn.Conv2d(branch_channels, branch_channels // 8, 1)
        self.dilation2_key = nn.Conv2d(branch_channels, branch_channels // 8, 1)
        self.dilation2_value = nn.Conv2d(branch_channels, branch_channels, 1)
        self.dilation2_scale = nn.Parameter(torch.zeros(1))

        self.dilated_branch4 = self._dilated_branch(channels, branch_channels, dilation=4)
        self.dilation4_query = nn.Conv2d(branch_channels, branch_channels // 8, 1)
        self.dilation4_key = nn.Conv2d(branch_channels, branch_channels // 8, 1)
        self.dilation4_value = nn.Conv2d(branch_channels, branch_channels, 1)
        self.dilation4_scale = nn.Parameter(torch.zeros(1))

        self.dilated_branch6 = self._dilated_branch(channels, branch_channels, dilation=6)
        self.dilation6_query = nn.Conv2d(branch_channels, branch_channels // 8, 1)
        self.dilation6_key = nn.Conv2d(branch_channels, branch_channels // 8, 1)
        self.dilation6_value = nn.Conv2d(branch_channels, branch_channels, 1)
        self.dilation6_scale = nn.Parameter(torch.zeros(1))

        self.strip_context = StripContextBranch(channels, branch_channels)
        self.branch_fusion = nn.Sequential(
            nn.Conv2d(5 * branch_channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
        )
        self.channel_recalibration = EfficientChannelAttention(channels)

    @staticmethod
    def _dilated_branch(in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                dilation=dilation,
                padding=dilation,
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    @staticmethod
    def _self_attention(feature, query_layer, key_layer, value_layer, scale):
        batch, channels, height, width = feature.shape
        query = query_layer(feature).view(batch, -1, height * width).permute(0, 2, 1)
        key = key_layer(feature).view(batch, -1, height * width)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        value = value_layer(feature).view(batch, -1, height * width)
        response = torch.bmm(value, attention.permute(0, 2, 1))
        response = response.view(batch, channels, height, width)
        return scale * response + feature

    def forward(self, x):
        x = self.input_projection(x)
        local = self.local_branch(x)
        dilation2 = self.dilated_branch2(x)
        dilation2 = self._self_attention(
            dilation2,
            self.dilation2_query,
            self.dilation2_key,
            self.dilation2_value,
            self.dilation2_scale,
        )
        dilation4 = self.dilated_branch4(x)
        dilation4 = self._self_attention(
            dilation4,
            self.dilation4_query,
            self.dilation4_key,
            self.dilation4_value,
            self.dilation4_scale,
        )
        dilation6 = self.dilated_branch6(x)
        dilation6 = self._self_attention(
            dilation6,
            self.dilation6_query,
            self.dilation6_key,
            self.dilation6_value,
            self.dilation6_scale,
        )
        strip = self.strip_context(x)
        fused = self.branch_fusion(
            torch.cat((local, dilation2, dilation4, dilation6, strip), dim=1)
        )
        return self.channel_recalibration(fused)


class DepthSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.mask_projection = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        maximum = torch.max(x, dim=1, keepdim=True)[0]
        return torch.sigmoid(self.mask_projection(maximum))


class DGAF(nn.Module):
    """Discrepancy-Guided Adaptive Fusion."""

    def __init__(self, channels):
        super().__init__()
        self.rgb_alignment = nn.Conv2d(channels, channels, 1, bias=False)
        self.depth_alignment = nn.Conv2d(channels, channels, 1, bias=False)
        self.discrepancy_estimator = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.depth_attention = DepthSpatialAttention(kernel_size=7)

    def forward(self, rgb, depth):
        aligned_rgb = self.rgb_alignment(rgb)
        aligned_depth = self.depth_alignment(depth)
        discrepancy = torch.abs(aligned_rgb - aligned_depth)
        reliability = 1.0 - self.discrepancy_estimator(discrepancy)
        depth_structure = self.depth_attention(aligned_depth)
        adaptive_guide = depth_structure * reliability
        enhanced_rgb = rgb + rgb * adaptive_guide
        filtered_depth = depth * reliability
        return torch.cat((enhanced_rgb, filtered_depth), dim=1)


class BoundaryRefinement(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gradient_path1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        # self.gradient_path1 = nn.Conv2d(
        #     channels, channels, kernel_size=(1, 3), padding=(0, 1),
        #     groups=channels, bias=False
        # )
        self.gradient_path2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        # self.gradient_path2 = nn.Conv2d(
        #     channels, channels, kernel_size=(3, 1), padding=(1, 0),
        #     groups=channels, bias=False
        # )
        self.gradient_norm = nn.BatchNorm2d(channels)
        self.output_refinement = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        gradient = torch.abs(self.gradient_path1(x)) + torch.abs(self.gradient_path2(x))
        attention = torch.sigmoid(self.gradient_norm(gradient))
        return self.output_refinement(x * attention + x)


class MMBE(nn.Module):
    """Multimodal Multi-scale Boundary Exploration."""

    def __init__(self, stage1_channels, stage2_channels, boundary_channels=32):
        super().__init__()
        self.rgb_stage1_projection = conv_bn_relu(stage1_channels, boundary_channels)
        self.rgb_stage2_projection = conv_bn_relu(stage2_channels, boundary_channels)
        self.rgb_scale_gate = nn.Sequential(
            nn.Conv2d(boundary_channels * 2, boundary_channels, 1, bias=False),
            nn.BatchNorm2d(boundary_channels),
            nn.Sigmoid(),
        )
        self.depth_stage1_projection = conv_bn_relu(stage1_channels, boundary_channels)
        self.depth_stage2_projection = conv_bn_relu(stage2_channels, boundary_channels)
        self.depth_scale_gate = nn.Sequential(
            nn.Conv2d(boundary_channels * 2, boundary_channels, 1, bias=False),
            nn.BatchNorm2d(boundary_channels),
            nn.Sigmoid(),
        )
        self.dgaf = DGAF(boundary_channels)
        self.boundary_refinement = BoundaryRefinement(boundary_channels * 2)
        self.boundary_head = nn.Conv2d(boundary_channels * 2, 1, 3, padding=1)

    @staticmethod
    def _combine_scales(stage1, stage2, stage1_projection, stage2_projection, gate):
        height, width = stage1.shape[2:]
        stage1 = stage1_projection(stage1)
        stage2 = stage2_projection(stage2)
        stage2 = F.interpolate(stage2, (height, width), mode="bilinear", align_corners=False)
        weight = gate(torch.cat((stage1, stage2), dim=1))
        return weight * stage1 + (1.0 - weight) * stage2

    def forward(self, rgb_stage1, rgb_stage2, depth_stage1, depth_stage2):
        rgb_boundary = self._combine_scales(
            rgb_stage1,
            rgb_stage2,
            self.rgb_stage1_projection,
            self.rgb_stage2_projection,
            self.rgb_scale_gate,
        )
        depth_boundary = self._combine_scales(
            depth_stage1,
            depth_stage2,
            self.depth_stage1_projection,
            self.depth_stage2_projection,
            self.depth_scale_gate,
        )
        boundary_feature = self.dgaf(rgb_boundary, depth_boundary)
        boundary_feature = self.boundary_refinement(boundary_feature)
        return self.boundary_head(boundary_feature), boundary_feature


class BoundaryChannelAttention(nn.Module):
    def __init__(self, channels, gamma=2, bias=1):
        super().__init__()
        kernel = int(abs(torch.log2(torch.tensor(channels)).item() / gamma + bias / gamma))
        kernel = kernel if kernel % 2 else kernel + 1
        kernel = max(kernel, 3)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_conv = nn.Conv1d(1, 1, kernel, padding=kernel // 2, bias=False)

    def forward(self, x):
        weights = self.global_pool(x).squeeze(-1).squeeze(-1).unsqueeze(1)
        weights = self.channel_conv(weights)
        weights = torch.sigmoid(weights).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        return x * weights


class BCFA(nn.Module):
    """Boundary-Guided Cross-layer Feature Aggregation."""

    def __init__(self, current_channels, deeper_channels, out_channels):
        super().__init__()
        self.cross_layer_projection = nn.Sequential(
            nn.Conv2d(current_channels + deeper_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.boundary_branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.boundary_channel_attention = BoundaryChannelAttention(out_channels)
        self.region_branch = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.output_refinement = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, boundary_prior, current_feature, deeper_feature):
        if deeper_feature.shape[2:] != current_feature.shape[2:]:
            deeper_feature = F.interpolate(
                deeper_feature,
                current_feature.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        mixed = self.cross_layer_projection(
            torch.cat((current_feature, deeper_feature), dim=1)
        )
        if boundary_prior.shape[2:] != mixed.shape[2:]:
            boundary_prior = F.interpolate(
                boundary_prior, mixed.shape[2:], mode="bilinear", align_corners=False
            )
        boundary_feature = self.boundary_branch(mixed * boundary_prior)
        boundary_feature = self.boundary_channel_attention(boundary_feature)
        region_feature = self.region_branch(mixed * (1.0 - boundary_prior))
        return self.output_refinement(boundary_feature + region_feature)


class RDFBDecoder(nn.Module):
    def __init__(self, stage4_channels, stage3_channels, stage2_channels, stage1_channels):
        super().__init__()
        self.bcfa_stage3 = BCFA(stage3_channels, stage4_channels, stage3_channels)
        self.bcfa_stage2 = BCFA(stage2_channels, stage3_channels, stage2_channels)
        self.bcfa_stage1 = BCFA(stage1_channels, stage2_channels, stage1_channels)

    def forward(self, boundary_prior, feature4, feature3, feature2, feature1):
        decoded3 = self.bcfa_stage3(boundary_prior, feature3, feature4)
        decoded2 = self.bcfa_stage2(boundary_prior, feature2, decoded3)
        decoded1 = self.bcfa_stage1(boundary_prior, feature1, decoded2)
        return decoded1, decoded3, decoded2


class RDFBNet(nn.Module):
    """RGB--Depth Fusion and Boundary-aware Network.

    Args:
        use_boundary_feature_fusion: Concatenate the MMBE boundary feature with
            the final decoder feature.
    """

    def __init__(self, use_boundary_feature_fusion=True):
        super().__init__()
        self.use_boundary_feature_fusion = use_boundary_feature_fusion
        channels = (64, 128, 320, 512)
        self.rgb_encoder = pvt_v2_b2()
        self.depth_encoder = pvt_v2_b1()
        self.cfer = CFER(channels[3])
        self.dgaf_stage4 = DGAF(channels[3])
        self.dgaf_stage3 = DGAF(channels[2])
        self.dgaf_stage2 = DGAF(channels[1])
        self.dgaf_stage1 = DGAF(channels[0])
        self.mmbe = MMBE(channels[0], channels[1], boundary_channels=32)

        if use_boundary_feature_fusion:
            self.boundary_feature_projection = conv_bn_relu(64, 32)

        fused_channels = tuple(channel * 2 for channel in channels)
        self.decoder = RDFBDecoder(
            fused_channels[3],
            fused_channels[2],
            fused_channels[1],
            fused_channels[0],
        )
        self.segmentation_head_stage4 = conv3x3(fused_channels[3], 1)
        self.segmentation_head_stage3 = conv3x3(fused_channels[2], 1)
        self.segmentation_head_stage2 = conv3x3(fused_channels[1], 1)
        self.final_feature_projection = conv_bn_relu(fused_channels[0], 32)
        final_channels = 64 if use_boundary_feature_fusion else 32
        self.final_segmentation_head = conv3x3(final_channels, 1)
        self.final_upsample = nn.UpsamplingBilinear2d(scale_factor=4)
        self.activation = nn.ReLU(False)

    def forward(self, rgb, pseudo_depth):
        rgb_stage1, rgb_stage2, rgb_stage3, rgb_stage4 = self.rgb_encoder(rgb)
        depth_stage1, depth_stage2, depth_stage3, depth_stage4 = self.depth_encoder(
            pseudo_depth
        )
        boundary_logit, boundary_feature = self.mmbe(
            rgb_stage1, rgb_stage2, depth_stage1, depth_stage2
        )
        boundary_prior = torch.sigmoid(boundary_logit).detach()

        rgb_stage4 = self.cfer(rgb_stage4)
        depth_stage4 = self.cfer(depth_stage4)
        feature4 = self.dgaf_stage4(rgb_stage4, depth_stage4)
        feature3 = self.dgaf_stage3(rgb_stage3, depth_stage3)
        feature2 = self.dgaf_stage2(rgb_stage2, depth_stage2)
        feature1 = self.dgaf_stage1(rgb_stage1, depth_stage1)
        decoded1, decoded3, decoded2 = self.decoder(
            boundary_prior, feature4, feature3, feature2, feature1
        )

        prediction4 = self.segmentation_head_stage4(feature4)
        prediction3 = self.segmentation_head_stage3(decoded3)
        prediction2 = self.segmentation_head_stage2(decoded2)
        final_feature = self.final_feature_projection(decoded1)

        if self.use_boundary_feature_fusion:
            projected_boundary = self.boundary_feature_projection(boundary_feature)
            final_feature = torch.cat((final_feature, projected_boundary), dim=1)

        final_feature = self.activation(final_feature)
        prediction1 = self.final_segmentation_head(self.final_upsample(final_feature))
        return prediction1, prediction2, prediction3, prediction4, boundary_logit
        # return prediction1, prediction2, prediction3, boundary_logit

    def load_pretrained_backbones(self, rgb_checkpoint, depth_checkpoint=None):
        rgb_state = _unwrap_checkpoint(torch.load(rgb_checkpoint, map_location="cpu"))
        depth_checkpoint = depth_checkpoint or rgb_checkpoint
        depth_state = _unwrap_checkpoint(torch.load(depth_checkpoint, map_location="cpu"))
        self.rgb_encoder.load_state_dict(rgb_state, strict=False)
        self.depth_encoder.load_state_dict(depth_state, strict=False)


def _unwrap_checkpoint(checkpoint):
    """Extract a state dict from common checkpoint wrappers."""
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a state dict or a dictionary containing one")
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            return candidate
    return checkpoint


__all__ = [
    "RDFBNet",
    "CFER",
    "DGAF",
    "MMBE",
    "BCFA",
    "RDFBDecoder",
]
