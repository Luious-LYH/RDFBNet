from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter


EPS = 1e-16


def load_grayscale(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def load_pair(prediction_path, ground_truth_path):
    prediction = Image.open(prediction_path).convert("L")
    ground_truth = Image.open(ground_truth_path).convert("L")
    if prediction.size != ground_truth.size:
        prediction = prediction.resize(ground_truth.size, Image.BILINEAR)
    return np.asarray(prediction), np.asarray(ground_truth)


def matched_pairs(prediction_dir, ground_truth_dir):
    prediction_dir = Path(prediction_dir)
    ground_truth_dir = Path(ground_truth_dir)
    prediction_files = {path.name: path for path in prediction_dir.iterdir() if path.is_file()}
    pairs = [
        (prediction_files[path.name], path)
        for path in sorted(ground_truth_dir.iterdir())
        if path.is_file() and path.name in prediction_files
    ]
    if not pairs:
        raise RuntimeError(
            f"No matching prediction/ground-truth filenames in {prediction_dir}"
        )
    return pairs


def prepare(prediction, ground_truth):
    prediction = np.asarray(prediction, dtype=np.float64) / 255.0
    ground_truth = np.asarray(ground_truth) > 128
    return prediction, ground_truth


def binary_overlap(prediction, ground_truth):
    binary_prediction = prediction > 0.5
    intersection = np.logical_and(binary_prediction, ground_truth).sum()
    prediction_area = binary_prediction.sum()
    ground_truth_area = ground_truth.sum()
    union = prediction_area + ground_truth_area
    dice = 2.0 * intersection / (union + EPS)
    iou = intersection / (union - intersection + EPS)
    mae = np.not_equal(binary_prediction, ground_truth).mean()
    return float(dice), float(iou), float(mae)


def _centroid(ground_truth, offset):
    height, width = ground_truth.shape
    if ground_truth.sum() == 0:
        return int(round(width / 2)) + offset, int(round(height / 2)) + offset
    area = ground_truth.sum()
    x = int(round(np.sum(ground_truth.sum(axis=0) * np.arange(width)) / area))
    y = int(round(np.sum(ground_truth.sum(axis=1) * np.arange(height)) / area))
    return x + offset, y + offset


def _ssim(prediction, ground_truth):
    size = prediction.size
    if size < 2:
        return 1.0
    prediction_mean = prediction.mean()
    ground_truth_mean = ground_truth.mean()
    prediction_variance = np.sum((prediction - prediction_mean) ** 2) / (size - 1)
    ground_truth_variance = np.sum((ground_truth - ground_truth_mean) ** 2) / (size - 1)
    covariance = np.sum(
        (prediction - prediction_mean) * (ground_truth - ground_truth_mean)
    ) / (size - 1)
    alpha = 4.0 * prediction_mean * ground_truth_mean * covariance
    beta = (prediction_mean**2 + ground_truth_mean**2) * (
        prediction_variance + ground_truth_variance
    )
    if alpha != 0:
        return alpha / (beta + EPS)
    return 1.0 if beta == 0 else 0.0


def _region_score(prediction, ground_truth, centroid_offset):
    x, y = _centroid(ground_truth, centroid_offset)
    height, width = ground_truth.shape
    prediction_regions = (
        prediction[:y, :x],
        prediction[:y, x:],
        prediction[y:, :x],
        prediction[y:, x:],
    )
    ground_truth_regions = (
        ground_truth[:y, :x],
        ground_truth[:y, x:],
        ground_truth[y:, :x],
        ground_truth[y:, x:],
    )
    weights = (
        x * y / (height * width),
        (width - x) * y / (height * width),
        x * (height - y) / (height * width),
        1.0 - x * y / (height * width)
        - (width - x) * y / (height * width)
        - x * (height - y) / (height * width),
    )
    return sum(
        weight * _ssim(prediction_region, ground_truth_region)
        for weight, prediction_region, ground_truth_region in zip(
            weights, prediction_regions, ground_truth_regions
        )
    )


def _object_score(prediction, ground_truth, sample_std):
    foreground = prediction[ground_truth]
    if foreground.size == 0:
        return 0.0
    ddof = 1 if sample_std and foreground.size > 1 else 0
    mean = foreground.mean()
    std = foreground.std(ddof=ddof)
    return 2.0 * mean / (mean**2 + 1.0 + std + EPS)


def s_measure(prediction, ground_truth, cod_protocol=False):
    foreground_ratio = ground_truth.mean()
    if foreground_ratio == 0:
        return float(1.0 - prediction.mean())
    if foreground_ratio == 1:
        return float(prediction.mean())

    foreground = prediction * ground_truth
    background = (1.0 - prediction) * np.logical_not(ground_truth)
    object_score = foreground_ratio * _object_score(
        foreground, ground_truth, cod_protocol
    )
    object_score += (1.0 - foreground_ratio) * _object_score(
        background, np.logical_not(ground_truth), cod_protocol
    )
    region_score = _region_score(
        prediction, ground_truth.astype(np.float64), 0 if cod_protocol else 1
    )
    return float(max(0.0, 0.5 * object_score + 0.5 * region_score))


def polyp_e_measure_curve(prediction, ground_truth):
    prediction_uint8 = (prediction * 255).astype(np.uint8)
    bins = np.linspace(0, 256, 257)
    foreground_histogram, _ = np.histogram(prediction_uint8[ground_truth], bins=bins)
    background_histogram, _ = np.histogram(
        prediction_uint8[np.logical_not(ground_truth)], bins=bins
    )
    foreground_foreground = np.cumsum(np.flip(foreground_histogram))
    foreground_background = np.cumsum(np.flip(background_histogram))
    predicted_foreground = foreground_foreground + foreground_background
    predicted_background = ground_truth.size - predicted_foreground
    ground_truth_foreground = np.count_nonzero(ground_truth)

    if ground_truth_foreground == 0:
        enhanced_sum = predicted_background
    elif ground_truth_foreground == ground_truth.size:
        enhanced_sum = predicted_foreground
    else:
        background_foreground = ground_truth_foreground - foreground_foreground
        background_background = predicted_background - background_foreground
        parts = (
            foreground_foreground,
            foreground_background,
            background_foreground,
            background_background,
        )
        mean_prediction = predicted_foreground / ground_truth.size
        mean_ground_truth = ground_truth_foreground / ground_truth.size
        combinations = (
            (1.0 - mean_prediction, 1.0 - mean_ground_truth),
            (1.0 - mean_prediction, -mean_ground_truth),
            (-mean_prediction, 1.0 - mean_ground_truth),
            (-mean_prediction, -mean_ground_truth),
        )
        enhanced_sum = np.zeros(256, dtype=np.float64)
        for part, (prediction_delta, ground_truth_delta) in zip(parts, combinations):
            alignment = 2.0 * prediction_delta * ground_truth_delta / (
                prediction_delta**2 + ground_truth_delta**2 + EPS
            )
            enhanced_sum += ((alignment + 1.0) ** 2 / 4.0) * part
    return enhanced_sum / (ground_truth.size - 1 + EPS)


def cod_mean_e_measure(prediction, ground_truth):
    scores = []
    for threshold in np.linspace(0.0, 1.0 - 1e-10, 255):
        binary_prediction = prediction >= threshold
        if ground_truth.mean() == 0:
            enhanced = np.logical_not(binary_prediction).astype(np.float64)
        elif ground_truth.mean() == 1:
            enhanced = binary_prediction.astype(np.float64)
        else:
            prediction_delta = binary_prediction - binary_prediction.mean()
            ground_truth_delta = ground_truth - ground_truth.mean()
            alignment = 2.0 * ground_truth_delta * prediction_delta / (
                ground_truth_delta**2 + prediction_delta**2 + 1e-20
            )
            enhanced = (alignment + 1.0) ** 2 / 4.0
        scores.append(enhanced.sum() / (ground_truth.size - 1 + 1e-20))
    return float(np.mean(scores))


def _gaussian_kernel(size=7, sigma=5):
    x, y = np.mgrid[
        -size // 2 + 1 : size // 2 + 1,
        -size // 2 + 1 : size // 2 + 1,
    ]
    kernel = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return kernel / kernel.sum()


def weighted_f_measure(prediction, ground_truth, cod_protocol=False):
    if not ground_truth.any():
        return float(1.0 - prediction.mean())

    background = np.logical_not(ground_truth)
    error = np.abs(prediction - ground_truth.astype(np.float64))
    distance, indices = distance_transform_edt(background, return_indices=True)
    propagated_error = error.copy()
    propagated_error[background] = error[
        indices[0, background], indices[1, background]
    ]
    if cod_protocol:
        smoothed_error = gaussian_filter(
            propagated_error,
            sigma=5.0,
            truncate=0.6,
            mode="constant",
            cval=0.0,
        )
    else:
        smoothed_error = convolve(
            propagated_error, _gaussian_kernel(), mode="nearest"
        )
    weighted_error = error.copy()
    foreground_update = np.logical_and(ground_truth, smoothed_error < error)
    weighted_error[foreground_update] = smoothed_error[foreground_update]
    importance = np.ones_like(error)
    importance[background] = 2.0 - np.exp(
        np.log(0.5) / 5.0 * distance[background]
    )
    weighted_error *= importance

    true_positive = ground_truth.sum() - weighted_error[ground_truth].sum()
    false_positive = weighted_error[background].sum()
    recall = 1.0 - weighted_error[ground_truth].mean()
    precision = true_positive / (true_positive + false_positive + np.finfo(float).eps)
    beta_squared = 0.3 if cod_protocol else 1.0
    score = (1.0 + beta_squared) * recall * precision
    score /= recall + beta_squared * precision + np.finfo(float).eps
    return float(score)
