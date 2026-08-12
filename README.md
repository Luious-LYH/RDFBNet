# RDFBNet

Official repository for **Pseudo-Depth-Assisted Polyp Segmentation via Discrepancy-Guided Fusion and Boundary-Guided Decoding**.

RDFBNet is a pseudo-depth-assisted segmentation model built around discrepancy-guided RGB-D fusion and boundary-guided decoding. In addition to polyp segmentation, we evaluate its cross-task applicability on camouflaged object detection (COD).

> The paper is currently under review. The remaining code will be released after acceptance.

## Architecture

<p align="center">
  <img src="assets/architecture.webp" alt="Overall architecture of RDFBNet" width="100%">
</p>

## Polyp Segmentation

### Quantitative Results

<p align="center">
  <img src="assets/polyp/quantitative.png" alt="Quantitative comparison on five polyp segmentation benchmarks" width="100%">
</p>

### Qualitative Results

<p align="center">
  <img src="assets/polyp/qualitative.webp" alt="Qualitative comparison on polyp segmentation benchmarks" width="100%">
</p>

The polyp segmentation prediction maps can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1kzv3rJYlhdnNGl3sVDfADHJ7mpuBNp8l?usp=drive_link).

## Camouflaged Object Detection

### Quantitative Results

<p align="center">
  <img src="assets/cod/quantitative.png" alt="Quantitative comparison on four COD benchmarks" width="100%">
</p>

### Qualitative Results

<p align="center">
  <img src="assets/cod/qualitative.webp" alt="Qualitative comparison on COD benchmarks" width="100%">
</p>

The COD prediction maps can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1XkS643AYDFOeCnF3r9rPPpsNfQEt0qWW?usp=drive_link).

## Evaluation

Polyp segmentation and COD use separate entry points so that each task follows the metric protocol reported in the paper.

```bash
python evaluation/evaluate_polyp.py --prediction-root /path/to/predictions --ground-truth-root /path/to/ground_truth
python evaluation/evaluate_cod.py --prediction-root /path/to/predictions --ground-truth-root /path/to/ground_truth
```

Each root directory should contain one subdirectory per dataset, with prediction and ground-truth images sharing the same filenames.

## Acknowledgements

We thank the authors of [PraNet](https://github.com/DengPingFan/PraNet), [PVT](https://github.com/whai362/PVT), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), and [SOD Evaluation Metrics](https://github.com/zyjwuyan/SOD_Evaluation_Metrics) for making their work publicly available.

## Citation

Citation information will be added when the paper's official bibliographic record becomes available.

## Contact

For questions, please contact the authors.
