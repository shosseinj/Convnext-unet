#!/usr/bin/env python3
"""Add color-coded contour overlays and extended panels to qualitative outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tools.generate_qualitative_ablation_examples import _panel


COLORS_RGB = {
    "GT": (0, 220, 0),
    "Baseline": (255, 50, 50),
    "Exp33": (0, 200, 255),
    "Exp45": (236, 64, 180),
}


def contours(mask_path: Path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Unable to read mask: {mask_path}")
    return cv2.findContours((mask >= 128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


def make_overlay(row: dict[str, str], destination: Path) -> None:
    original = cv2.imread(row["original_path"], cv2.IMREAD_COLOR)
    if original is None:
        raise RuntimeError(f"Unable to read original: {row['original_path']}")
    sample_dir = Path(row["original_path"]).parent
    sources = {
        "GT": sample_dir / "ground_truth_mask.png",
        "Baseline": Path(row["baseline_mask_path"]),
        "Exp33": Path(row["second_ablation_mask_path"]),
        "Exp45": Path(row["third_ablation_mask_path"]),
    }
    thickness = max(2, round(min(original.shape[:2]) / 220))
    for label, mask_path in sources.items():
        rgb = COLORS_RGB[label]
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.drawContours(original, contours(mask_path), -1, bgr, thickness, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(original.shape[:2]) / 700)
    line_height = max(22, round(32 * scale))
    legend_width = max(150, round(180 * scale))
    legend_height = line_height * len(sources) + 12
    overlay = original.copy()
    cv2.rectangle(overlay, (8, 8), (8 + legend_width, 8 + legend_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, original, 0.38, 0, original)
    for index, (label, rgb) in enumerate(COLORS_RGB.items()):
        y = 8 + line_height * index + line_height - 7
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.line(original, (18, y - 4), (45, y - 4), bgr, max(2, thickness), cv2.LINE_AA)
        cv2.putText(original, label, (54, y), font, scale, (255, 255, 255), max(1, thickness // 2), cv2.LINE_AA)
    rgb_image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb_image).save(destination, format="PNG", compress_level=1, dpi=(300, 300))


def main() -> int:
    root = Path("qualitative_ablation_outputs")
    manifest_path = root / "manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    extra_fields = ["contour_overlay_path", "comparison_panel_with_gt_and_contours_path"]
    for field in extra_fields:
        if field not in fields:
            fields.append(field)
    for row in rows:
        sample_dir = Path(row["original_path"]).parent
        overlay_path = sample_dir / "contour_overlay.png"
        panel_path = sample_dir / "comparison_panel_with_gt_and_contours.png"
        make_overlay(row, overlay_path)
        _panel(
            (Path(row["original_path"]), sample_dir / "ground_truth_mask.png",
             Path(row["baseline_mask_path"]), Path(row["second_ablation_mask_path"]),
             Path(row["third_ablation_mask_path"]), overlay_path),
            ("Original", "GT", "Baseline", "Second Ablation", "Third Ablation", "Contour Overlay"),
            panel_path,
        )
        row["contour_overlay_path"] = str(overlay_path.resolve())
        row["comparison_panel_with_gt_and_contours_path"] = str(panel_path.resolve())
        print(f"Created {panel_path}")
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Added {len(rows)} contour overlays and {len(rows)} extended panels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
