"""Evaluate polyp segmentation prediction maps."""

import argparse
from pathlib import Path

import numpy as np

try:
    from .metrics import (
        binary_overlap,
        load_pair,
        matched_pairs,
        polyp_e_measure_curve,
        prepare,
        s_measure,
        weighted_f_measure,
    )
except ImportError:
    from metrics import (
        binary_overlap,
        load_pair,
        matched_pairs,
        polyp_e_measure_curve,
        prepare,
        s_measure,
        weighted_f_measure,
    )


def evaluate_dataset(prediction_dir, ground_truth_dir):
    dice_scores, iou_scores, mae_scores = [], [], []
    s_scores, e_curves, weighted_f_scores = [], [], []
    for prediction_path, ground_truth_path in matched_pairs(
        prediction_dir, ground_truth_dir
    ):
        prediction, ground_truth = prepare(
            *load_pair(prediction_path, ground_truth_path)
        )
        dice, iou, mae = binary_overlap(prediction, ground_truth)
        dice_scores.append(dice)
        iou_scores.append(iou)
        mae_scores.append(mae)
        s_scores.append(s_measure(prediction, ground_truth))
        e_curves.append(polyp_e_measure_curve(prediction, ground_truth))
        weighted_f_scores.append(weighted_f_measure(prediction, ground_truth))
    return {
        "mDice": float(np.mean(dice_scores)),
        "mIoU": float(np.mean(iou_scores)),
        "Sm": float(np.mean(s_scores)),
        "maxEm": float(np.mean(e_curves, axis=0).max()),
        "wFm": float(np.mean(weighted_f_scores)),
        "MAE": float(np.mean(mae_scores)),
    }


def dataset_names(prediction_root, ground_truth_root, requested):
    if requested:
        return requested
    return sorted(
        path.name
        for path in ground_truth_root.iterdir()
        if path.is_dir() and (prediction_root / path.name).is_dir()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()

    datasets = dataset_names(
        args.prediction_root, args.ground_truth_root, args.datasets
    )
    if not datasets:
        raise RuntimeError("No dataset directories were found")
    for dataset in datasets:
        results = evaluate_dataset(
            args.prediction_root / dataset, args.ground_truth_root / dataset
        )
        values = "  ".join(f"{name}: {value:.4f}" for name, value in results.items())
        print(f"{dataset}  {values}")


if __name__ == "__main__":
    main()
