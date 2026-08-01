#!/usr/bin/env python3
"""Aggregate verified official evaluation summaries across the three fixed seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 3407, 2026)
DATASETS = ("Kvasir-SEG", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-LARIBPOLYPDB")
METRICS = ("dice", "iou", "precision", "recall", "specificity", "pixel_accuracy", "mae")
T_CRITICAL_95_DF2 = 4.302652729911275


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("Expected exactly three finite seed values")
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    half_width = float(T_CRITICAL_95_DF2 * std / math.sqrt(3))
    return {"mean": mean, "sample_std": std,
            "ci95_lower": mean - half_width, "ci95_upper": mean + half_width,
            "values": {str(seed): float(value) for seed, value in zip(SEEDS, array)}}


def load_seed_summaries(results_root, variant, dataset):
    rows = []
    for seed in SEEDS:
        path = results_root / "evaluation" / variant / f"seed_{seed}" / dataset / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing official evaluation summary: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "PASS" or value.get("seed") != seed or value.get("variant") != variant:
            raise ValueError(f"Invalid evaluation identity/status: {path}")
        if value.get("dataset") != dataset or value.get("threshold") != 0.5 or value.get("tta") is not False:
            raise ValueError(f"Evaluation protocol mismatch: {path}")
        value["_source_path"] = str(path)
        value["_source_sha256"] = sha256_file(path)
        rows.append(value)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    output = {"status": "PASS", "variant": args.variant, "seeds": list(SEEDS),
              "method": {"standard_deviation": "sample", "ci95": "two-sided Student t, df=2",
                         "t_critical": T_CRITICAL_95_DF2},
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "source_summaries": [], "datasets": {}}
    flat_rows = []
    for dataset in DATASETS:
        seed_rows = load_seed_summaries(args.results_root, args.variant, dataset)
        output["source_summaries"].extend(
            {"dataset": dataset, "seed": row["seed"], "path": row["_source_path"],
             "sha256": row["_source_sha256"]} for row in seed_rows
        )
        aggregated = {metric: summarize([row["metrics"][metric] for row in seed_rows])
                      for metric in METRICS}
        size_labels = sorted(set.intersection(*[set(row["size_stratified"]) for row in seed_rows]))
        size_aggregated = {
            label: {metric: summarize([row["size_stratified"][label][metric] for row in seed_rows])
                    for metric in METRICS}
            for label in size_labels
        }
        output["datasets"][dataset] = {"metrics": aggregated, "size_stratified": size_aggregated}
        for metric, stats in aggregated.items():
            flat_rows.append({"variant": args.variant, "dataset": dataset, "metric": metric,
                              "seed_42": stats["values"]["42"],
                              "seed_3407": stats["values"]["3407"],
                              "seed_2026": stats["values"]["2026"],
                              "mean": stats["mean"], "sample_std": stats["sample_std"],
                              "ci95_lower": stats["ci95_lower"], "ci95_upper": stats["ci95_upper"]})
    output_dir = args.results_root / "aggregated" / "official"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{args.variant}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (output_dir / f"{args.variant}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_rows[0].keys())
        writer.writeheader(); writer.writerows(flat_rows)
    print(f"Wrote validated aggregate for {args.variant}: {output_dir}")


if __name__ == "__main__":
    main()
