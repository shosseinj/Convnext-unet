"""Deterministic per-image segmentation metrics for official evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SizeBins:
    small_max: float = 0.05
    medium_max: float = 0.20

    def label(self, area_ratio: float) -> str:
        if area_ratio < self.small_max:
            return "small"
        if area_ratio <= self.medium_max:
            return "medium"
        return "large"


def binary_metrics(probability: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """Return one metric record per image with explicit empty-set semantics."""
    if probability.shape != target.shape or probability.ndim != 4:
        raise ValueError("probability and target must have identical NCHW shapes")
    prediction = probability >= threshold
    truth = target >= 0.5
    dims = (1, 2, 3)
    tp = (prediction & truth).sum(dims).double()
    fp = (prediction & ~truth).sum(dims).double()
    fn = (~prediction & truth).sum(dims).double()
    tn = (~prediction & ~truth).sum(dims).double()

    def safe_ratio(numerator, denominator, empty_value=1.0):
        fallback = torch.full_like(denominator, float(empty_value))
        return torch.where(denominator > 0, numerator / denominator, fallback)

    dice = safe_ratio(2 * tp, 2 * tp + fp + fn)
    iou = safe_ratio(tp, tp + fp + fn)
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    accuracy = safe_ratio(tp + tn, tp + tn + fp + fn)
    mae = torch.abs(probability.double() - target.double()).mean(dims)
    area_ratio = truth.double().mean(dims)
    records = []
    for index in range(probability.shape[0]):
        records.append({
            "dice": float(dice[index]),
            "iou": float(iou[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "specificity": float(specificity[index]),
            "pixel_accuracy": float(accuracy[index]),
            "mae": float(mae[index]),
            "mask_area_ratio": float(area_ratio[index]),
            "ground_truth_positive_pixels": int((tp + fn)[index]),
            "predicted_positive_pixels": int((tp + fp)[index]),
        })
    return records


def aggregate_records(records: list[dict], metric_names: tuple[str, ...]):
    if not records:
        raise ValueError("Cannot aggregate an empty metric record list")
    result = {name: float(np.mean([row[name] for row in records])) for name in metric_names}
    result["samples"] = len(records)
    return result
