"""Compatibility helpers for the paper qualitative-figure tests."""

import numpy as np

from tools.generate_qualitative_ablation_examples import deterministic_selection


def colorize_masks(truth: np.ndarray, prediction: np.ndarray):
    """Return RGB ground truth, magenta prediction, and TP/FP/FN error maps."""
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if truth.shape != prediction.shape:
        raise ValueError("Ground truth and prediction shapes must match")
    ground_truth = np.zeros((*truth.shape, 3), dtype=np.uint8)
    ground_truth[truth] = (255, 255, 255)
    predicted = np.zeros_like(ground_truth)
    predicted[prediction] = (236, 64, 180)
    error = np.zeros_like(ground_truth)
    error[truth & prediction] = (255, 255, 255)
    error[~truth & prediction] = (255, 215, 0)
    error[truth & ~prediction] = (0, 206, 209)
    return ground_truth, predicted, error
