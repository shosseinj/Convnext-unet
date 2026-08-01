"""Reusable helpers for video segmentation inference and evaluation."""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def preprocess_frame(frame_bgr: np.ndarray, input_size: int) -> torch.Tensor:
    """Match repository validation preprocessing: BGR->RGB, resize, [0,1], CHW."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    array = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1))).unsqueeze(0)


def logits_to_probability(logits: torch.Tensor) -> torch.Tensor:
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if logits.ndim != 4:
        raise ValueError(f"Expected BCHW model output, got shape {tuple(logits.shape)}")
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)[:, 0]
    if logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    raise ValueError(f"Expected 1 or 2 output channels, got {logits.shape[1]}")


def resize_probability(probability: torch.Tensor, height: int, width: int) -> np.ndarray:
    if probability.ndim == 2:
        probability = probability.unsqueeze(0).unsqueeze(0)
    elif probability.ndim == 3:
        probability = probability.unsqueeze(1)
    resized = F.interpolate(probability, (height, width), mode="bilinear", align_corners=False)
    return resized[0, 0].detach().cpu().numpy()


def filter_components(mask: np.ndarray, min_area: int) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)
    components: List[Dict[str, int]] = []
    for label in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label])
        if area < min_area:
            continue
        filtered[labels == label] = 1
        components.append({"label": label, "x": x, "y": y, "width": width, "height": height, "area": area})
    return filtered, components


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = prediction.astype(bool)
    truth = target.astype(bool)
    tp = int(np.logical_and(pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())

    def ratio(numerator: int, denominator: int, empty_value: float = 1.0) -> float:
        return float(numerator / denominator) if denominator else empty_value

    return {
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "pixel_accuracy": ratio(tp + tn, tp + tn + fp + fn),
    }


def annotate_prediction(frame: np.ndarray, probability: np.ndarray, mask: np.ndarray,
                        components: List[Dict[str, int]], alpha: float,
                        contour_thickness: int, box_thickness: int) -> np.ndarray:
    result = frame.copy()
    color_layer = np.zeros_like(result)
    color_layer[mask.astype(bool)] = (0, 0, 255)
    result = cv2.addWeighted(result, 1.0, color_layer, alpha, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 255), contour_thickness)
    for item in components:
        x, y, width, height = item["x"], item["y"], item["width"], item["height"]
        component_pixels = mask[y:y + height, x:x + width].astype(bool)
        component_probs = probability[y:y + height, x:x + width][component_pixels]
        confidence = float(component_probs.mean()) if component_probs.size else 0.0
        cv2.rectangle(result, (x, y), (x + width - 1, y + height - 1), (0, 255, 0), box_thickness)
        cv2.putText(result, f"Polyp {confidence:.2f}", (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    return result
