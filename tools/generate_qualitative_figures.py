#!/usr/bin/env python3
"""Generate deterministic qualitative panels from a validated full-model evaluation."""

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.architecture_factory import build_variant_by_id
from research_pipeline.data import ManifestSegmentationDataset
from research_pipeline.reproducibility import list_pairs
from tools.evaluate_official import dataset_pairs, sha256_file
from tools.run_official_queue import validate_completed_run


SIZE_ORDER = ("small", "medium", "large")
DATASETS = ("Kvasir-SEG", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-LARIBPOLYPDB")


def deterministic_selection(rows):
    selected = []
    for label in SIZE_ORDER:
        candidates = [row for row in rows if row["size_bin"] == label]
        if not candidates:
            continue
        candidates.sort(key=lambda row: (float(row["mask_area_ratio"]), row["image"]))
        median = float(np.median([float(row["mask_area_ratio"]) for row in candidates]))
        chosen = min(candidates, key=lambda row: (abs(float(row["mask_area_ratio"]) - median), row["image"]))
        selected.append({"size_bin": label, "image": chosen["image"],
                         "mask_area_ratio": float(chosen["mask_area_ratio"]),
                         "selection_median_area_ratio": median})
    return selected


def colorize_masks(truth, prediction):
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    ground_truth = np.zeros((*truth.shape, 3), dtype=np.uint8)
    ground_truth[truth] = (255, 255, 255)
    predicted = np.zeros((*truth.shape, 3), dtype=np.uint8)
    predicted[prediction] = (236, 64, 180)  # magenta/pink
    error = np.zeros((*truth.shape, 3), dtype=np.uint8)
    error[truth & prediction] = (255, 255, 255)  # true positive
    error[~truth & prediction] = (255, 215, 0)   # false positive: yellow
    error[truth & ~prediction] = (0, 206, 209)   # false negative: cyan
    return ground_truth, predicted, error


def add_header(image, text, height=34):
    canvas = Image.new("RGB", (image.width, image.height + height), "white")
    canvas.paste(image, (0, height))
    ImageDraw.Draw(canvas).text((8, 9), text, fill="black")
    return canvas


def render_dataset(dataset_name, records, panels, output_path):
    cell = 352
    header = 34
    columns = ("Input", "Ground truth", "Prediction (pink)", "Error: FP yellow, FN cyan")
    canvas = Image.new("RGB", (4 * cell, len(records) * (cell + header)), "white")
    for row_index, record in enumerate(records):
        images = panels[record["image"]]
        for column, (label, array) in enumerate(zip(columns, images)):
            panel = add_header(Image.fromarray(array), f"{record['size_bin']} | {label}", header)
            canvas.paste(panel, (column * cell, row_index * (cell + header)))
    canvas.save(output_path, format="PNG")
    return {"dataset": dataset_name, "path": str(output_path), "rows": len(records),
            "sha256": sha256_file(output_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "manuscript" / "figures")
    args = parser.parse_args()
    variant, seed = "full", 42
    run_dir = args.results_root / "raw" / variant / f"seed_{seed}" / "official"
    valid, reason = validate_completed_run(run_dir, variant, seed)
    validation_path = run_dir / "validation.json"
    if not valid or not validation_path.is_file():
        raise SystemExit(f"Qualitative generation BLOCKED: full/seed42 is not validated ({reason})")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise SystemExit("Qualitative generation BLOCKED: deep validation is not PASS")

    checkpoint_path = run_dir / "best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest_path = ROOT / "configs" / "splits" / "development_seed_42.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = yaml.safe_load((ROOT / "configs" / "training_protocol.yaml").read_text(encoding="utf-8"))
    pretrained = checkpoint.get("metadata", {}).get("pretrained_weights", {}).get("path")
    device = torch.device(args.device)
    model = build_variant_by_id(variant, weights_path=pretrained).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selections, figures = [], []

    for dataset_name in DATASETS:
        per_image = args.results_root / "evaluation" / variant / f"seed_{seed}" / dataset_name / "per_image.csv"
        summary_path = per_image.with_name("summary.json")
        if not per_image.is_file() or not summary_path.is_file():
            raise SystemExit(f"Qualitative generation BLOCKED: missing evaluation evidence for {dataset_name}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (summary.get("status") != "PASS" or summary.get("threshold") != 0.5
                or summary.get("tta") is not False
                or summary.get("checkpoint_sha256") != sha256_file(checkpoint_path)):
            raise SystemExit(f"Qualitative generation BLOCKED: evaluation contract mismatch for {dataset_name}")
        with per_image.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        selected = deterministic_selection(rows)
        pairs, split = dataset_pairs(dataset_name, manifest, args.data_root)
        pair_by_image = {pair["image"]: pair for pair in pairs}
        dataset = ManifestSegmentationDataset(args.data_root, dataset_name, [], protocol["input_size"][0], False)
        panels = {}
        for record in selected:
            if record["image"] not in pair_by_image:
                raise RuntimeError(f"Selected image is absent from fixed split: {dataset_name}/{record['image']}")
            dataset.pairs = [pair_by_image[record["image"]]]
            image, mask, _ = dataset[0]
            with torch.inference_mode():
                output = model(image.unsqueeze(0).to(device))
                output = output[0] if isinstance(output, (tuple, list)) else output
                prediction = (torch.sigmoid(output)[0, 0].cpu().numpy() >= 0.5)
            truth = mask[0].numpy() >= 0.5
            input_rgb = (image.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
            ground_truth, predicted, error = colorize_masks(truth, prediction)
            panels[record["image"]] = (input_rgb, ground_truth, predicted, error)
            selections.append({"dataset": dataset_name, "split": split, **record})
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", dataset_name).strip("_").lower()
        output_path = args.output_dir / f"qualitative_{safe_name}.png"
        figures.append(render_dataset(dataset_name, selected, panels, output_path))

    evidence = {"status": "PASS", "selection_rule": "closest ground-truth area ratio to the median within each size bin; filename breaks ties",
                "selection_uses_prediction_performance": False, "variant": variant, "seed": seed,
                "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
                "threshold": 0.5, "tta": False, "colors": {"prediction": "pink/magenta",
                "true_positive": "white", "false_positive": "yellow", "false_negative": "cyan"},
                "selections": selections, "figures": figures,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    evidence_path = args.output_dir / "qualitative_selection.json"
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    caption = ROOT / "manuscript" / "captions" / "qualitative.md"
    caption.write_text(
        "**Figure X. Deterministically selected qualitative segmentation examples.** For each dataset and available "
        "ground-truth size bin, the displayed image has the area ratio closest to the bin median; filename order "
        "breaks ties, and prediction performance is not used for selection. Predictions are pink/magenta. In the "
        "error map, true-positive pixels are white, false-positive pixels are yellow, false-negative pixels are "
        "cyan, and true-negative pixels are black. The checkpoint, threshold, split, and selected filenames are "
        "recorded in `manuscript/figures/qualitative_selection.json`.\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "figures": len(figures), "selections": len(selections)}))


if __name__ == "__main__":
    main()
