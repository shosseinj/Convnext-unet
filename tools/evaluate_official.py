#!/usr/bin/env python3
"""Evaluate one completed official checkpoint without selection or tuning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.architecture_factory import build_variant_by_id
from research_pipeline.data import ManifestSegmentationDataset
from research_pipeline.evaluation import SizeBins, aggregate_records, binary_metrics
from research_pipeline.reproducibility import list_pairs
from tools.run_official_queue import validate_completed_run


METRICS = ("dice", "iou", "precision", "recall", "specificity", "pixel_accuracy", "mae")
EXTERNAL_TESTS = ("CVC-300", "CVC-ColonDB", "ETIS-LARIBPOLYPDB")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_pairs(name, manifest, data_root):
    if name in ("Kvasir-SEG", "CVC-ClinicDB"):
        return manifest["datasets"][name]["validation"], "validation"
    if name not in EXTERNAL_TESTS:
        raise ValueError(f"Dataset is not in the fixed evaluation protocol: {name}")
    return list_pairs(data_root / name), "external_test"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(42, 3407, 2026))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--threshold", type=float, default=0.5, choices=(0.5,))
    args = parser.parse_args()

    run_dir = args.results_root / "raw" / args.variant / f"seed_{args.seed}" / "official"
    valid, reason = validate_completed_run(run_dir, args.variant, args.seed)
    if not valid:
        raise SystemExit(f"Official run is not eligible for evaluation: {reason}")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest_path = ROOT / "configs" / "splits" / f"development_seed_{args.seed}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = yaml.safe_load((ROOT / "configs" / "training_protocol.yaml").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    pretrained = checkpoint.get("metadata", {}).get("pretrained_weights", {}).get("path")
    model = build_variant_by_id(args.variant, weights_path=pretrained).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    bins_cfg = protocol["reporting"]["size_bins"]
    bins = SizeBins(0.05, 0.20)

    for dataset_name in (*manifest["datasets"].keys(), *EXTERNAL_TESTS):
        pairs, split = dataset_pairs(dataset_name, manifest, args.data_root)
        dataset = ManifestSegmentationDataset(
            args.data_root, dataset_name, pairs, protocol["input_size"][0], training=False
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        rows = []
        with torch.inference_mode():
            for images, masks, filenames in loader:
                output = model(images.to(device))
                output = output[0] if isinstance(output, (tuple, list)) else output
                probabilities = torch.sigmoid(output).cpu()
                batch_records = binary_metrics(probabilities, masks, args.threshold)
                for filename, record in zip(filenames, batch_records):
                    record.update({"image": filename, "size_bin": bins.label(record["mask_area_ratio"])})
                    rows.append(record)
        if len(rows) != len(pairs) or not rows:
            raise RuntimeError(f"Evaluation row count mismatch for {dataset_name}")
        output_dir = args.results_root / "evaluation" / args.variant / f"seed_{args.seed}" / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = ("image", *METRICS, "mask_area_ratio", "size_bin",
                      "ground_truth_positive_pixels", "predicted_positive_pixels")
        with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(rows)
        size_summary = {
            label: aggregate_records([row for row in rows if row["size_bin"] == label], METRICS)
            for label in ("small", "medium", "large") if any(row["size_bin"] == label for row in rows)
        }
        result = {
            "status": "PASS",
            "variant": args.variant,
            "seed": args.seed,
            "dataset": dataset_name,
            "split": split,
            "threshold": args.threshold,
            "tta": False,
            "metrics": aggregate_records(rows, METRICS),
            "size_stratified": size_summary,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_selection_dice": summary["best_selection_dice"],
            "manifest": str(manifest_path),
            "manifest_sha256": manifest["sha256"],
            "evaluation_code_sha256": sha256_file(Path(__file__)),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "size_bin_protocol": bins_cfg,
        }
        (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"dataset": dataset_name, "metrics": result["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
