#!/usr/bin/env python3
"""Audit qualitative output counts and report per-model Dice on selected samples."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image


root = Path("qualitative_ablation_outputs")
with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
scores = {"baseline": [], "exp33": [], "exp45": []}
columns = {
    "baseline": "baseline_mask_path",
    "exp33": "second_ablation_mask_path",
    "exp45": "third_ablation_mask_path",
}
print("dataset,image_id,baseline_dice,exp33_dice,exp45_dice,best")
for row in rows:
    sample_dir = Path(row["original_path"]).parent
    truth = np.asarray(Image.open(sample_dir / "ground_truth_mask.png")) >= 128
    row_scores = {}
    for model, column in columns.items():
        prediction = np.asarray(Image.open(row[column])) >= 128
        intersection = np.logical_and(prediction, truth).sum()
        row_scores[model] = float((2 * intersection + 1e-6) / (prediction.sum() + truth.sum() + 1e-6))
        scores[model].append(row_scores[model])
    best = max(row_scores, key=row_scores.get)
    print(f"{row['dataset']},{row['image_id']},{row_scores['baseline']:.6f},"
          f"{row_scores['exp33']:.6f},{row_scores['exp45']:.6f},{best}")
print("means," + ",".join(f"{model}={np.mean(values):.6f}" for model, values in scores.items()))
print(f"exp45_wins,{sum(t >= max(b, s) for b, s, t in zip(scores['baseline'], scores['exp33'], scores['exp45']))}/{len(rows)}")
