"""Evaluate camouflaged object detection prediction maps."""

import argparse
from pathlib import Path

import numpy as np

try:
    from .metrics import (
        cod_mean_e_measure,
        load_pair,
        matched_pairs,
        s_measure,
        weighted_f_measure,
    )
except ImportError:
    from metrics import (
        cod_mean_e_measure,
        load_pair,
        matched_pairs,
        s_measure,
        weighted_f_measure,
    )


def evaluate_dataset(prediction_dir, ground_truth_dir):
    s_scores, mean_e_scores, weighted_f_scores, mae_scores = [], [], [], []
    for prediction_path, ground_truth_path in matched_pairs(
        prediction_dir, ground_truth_dir
    ):
        prediction_uint8, ground_truth_uint8 = load_pair(
            prediction_path, ground_truth_path
        )
        prediction = prediction_uint8.astype(np.float64) / 255.0
        ground_truth_float = ground_truth_uint8.astype(np.float64) / 255.0
        ground_truth = ground_truth_uint8 > 128
        s_scores.append(s_measure(prediction, ground_truth, cod_protocol=True))
        mean_e_scores.append(cod_mean_e_measure(prediction, ground_truth))
        weighted_f_scores.append(
            weighted_f_measure(prediction, ground_truth, cod_protocol=True)
        )
        mae_scores.append(np.abs(prediction - ground_truth_float).mean())
    return {
        "Sm": float(np.mean(s_scores)),
        "meanEm": float(np.mean(mean_e_scores)),
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
