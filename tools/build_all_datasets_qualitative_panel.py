#!/usr/bin/env python3
"""Build one journal-ready qualitative comparison plate across all datasets."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("qualitative_ablation_outputs")
OUTPUT = ROOT / "all_datasets_comparison_panel_with_gt.png"
COLUMNS = (
    ("Original", "original_path"),
    ("Ground Truth", "ground_truth_mask.png"),
    ("Baseline (Exp01)", "baseline_mask_path"),
    ("Second Ablation (Exp33)", "second_ablation_mask_path"),
    ("Third Ablation (Exp45)", "third_ablation_mask_path"),
)


def font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def fit(image: Image.Image, width: int, height: int) -> Image.Image:
    is_mask = image.mode in {"1", "L", "I", "I;16"}
    copy = image.convert("RGB")
    scale = min(width / copy.width, height / copy.height)
    resized_size = (max(1, round(copy.width * scale)), max(1, round(copy.height * scale)))
    resampling = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS
    resized = copy.resize(resized_size, resampling)
    cell = Image.new("RGB", (width, height), "black")
    cell.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    resized.close()
    return cell


def centered_text(draw, box, text, text_font, fill=(20, 20, 20)):
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=text_font)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.text((left + (right - left - width) / 2, top + (bottom - top - height) / 2),
              text, font=text_font, fill=fill)


def main() -> int:
    with (ROOT / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 10:
        raise RuntimeError(f"Expected 10 manifest rows, found {len(rows)}")

    label_width, cell_width, cell_height = 390, 700, 525
    outer, gap, header_height = 55, 22, 115
    row_height = cell_height + 48
    width = outer * 2 + label_width + len(COLUMNS) * cell_width + (len(COLUMNS) - 1) * gap
    height = outer * 2 + header_height + len(rows) * row_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font, dataset_font, sample_font = font(34, True), font(32, True), font(25)

    centered_text(draw, (outer, outer, outer + label_width, outer + header_height),
                  "Dataset / Sample", header_font)
    for index, (title, _) in enumerate(COLUMNS):
        x = outer + label_width + index * (cell_width + gap)
        centered_text(draw, (x, outer, x + cell_width, outer + header_height), title, header_font)
    draw.line((outer, outer + header_height, width - outer, outer + header_height), fill=(40, 40, 40), width=4)

    previous_dataset = None
    for row_index, row in enumerate(rows):
        row_top = outer + header_height + row_index * row_height
        dataset = row["dataset"]
        if dataset != previous_dataset:
            if row_index:
                draw.line((outer, row_top, width - outer, row_top), fill=(55, 55, 55), width=5)
            previous_dataset = dataset
        if row_index % 2:
            draw.rectangle((outer, row_top, width - outer, row_top + row_height), fill=(246, 247, 249))
        label_center = row_top + row_height / 2
        dataset_bounds = draw.textbbox((0, 0), dataset, font=dataset_font)
        dataset_width = dataset_bounds[2] - dataset_bounds[0]
        draw.text((outer + (label_width - dataset_width) / 2, label_center - 42), dataset,
                  font=dataset_font, fill=(15, 15, 15))
        sample = f"Sample {int(row['sample_index']):02d}  |  {row['image_id']}"
        sample_bounds = draw.textbbox((0, 0), sample, font=sample_font)
        sample_width = sample_bounds[2] - sample_bounds[0]
        draw.text((outer + (label_width - sample_width) / 2, label_center + 8), sample,
                  font=sample_font, fill=(75, 75, 75))

        sample_dir = Path(row["original_path"]).parent
        for column_index, (_, source) in enumerate(COLUMNS):
            path = sample_dir / source if source.endswith(".png") and source not in row else Path(row[source])
            with Image.open(path) as image:
                prepared = fit(image, cell_width, cell_height)
            x = outer + label_width + column_index * (cell_width + gap)
            y = row_top + (row_height - prepared.height) // 2
            canvas.paste(prepared, (x + (cell_width - prepared.width) // 2, y))
            prepared.close()

    canvas.save(OUTPUT, format="PNG", compress_level=1, dpi=(300, 300))
    print(f"Saved {OUTPUT.resolve()}")
    print(f"Canvas: {width} x {height} pixels at 300 DPI; rows={len(rows)}; datasets=5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
