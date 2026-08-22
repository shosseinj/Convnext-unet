"""Aggregate three valid ablation evaluation summaries into summary.csv."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from ablation_artifacts import (
    COMPLEXITY_FIELDS, DATASETS, METRICS, checkpoint_sha256,
    validate_evaluation_summary,
)
from ablation_registry import canonical_seeds


def summarize(experiment_dir, experiment_name, validate_fingerprint=True):
    experiment_dir = Path(experiment_dir)
    payloads = []
    for seed in canonical_seeds():
        seed_dir = experiment_dir / f"seed_{seed}"
        summary_path = seed_dir / "evaluation_summary.json"
        fingerprint = None
        if validate_fingerprint:
            checkpoint_path = seed_dir / "best_checkpoint.pth"
            if not checkpoint_path.is_file():
                raise RuntimeError(f"Missing checkpoint for seed {seed}: {checkpoint_path}")
            fingerprint = checkpoint_sha256(checkpoint_path)
        validation = validate_evaluation_summary(
            summary_path, experiment_name, seed, fingerprint
        )
        if not validation.valid:
            raise RuntimeError(f"Invalid evaluation for seed {seed}: {validation.reason}")
        payloads.append(json.loads(summary_path.read_text(encoding="utf-8")))

    rows = []
    for dataset in DATASETS:
        for metric in METRICS:
            values = [float(payload["results"][dataset][metric]) for payload in payloads]
            rows.append({
                "row_type": "metric", "dataset": dataset, "name": metric,
                "mean": statistics.mean(values), "sample_std": statistics.stdev(values),
            })
    for field in COMPLEXITY_FIELDS:
        values = [float(payload[field]) for payload in payloads]
        rows.append({
            "row_type": "complexity", "dataset": "", "name": field,
            "mean": statistics.mean(values), "sample_std": statistics.stdev(values),
        })

    output = experiment_dir / "summary.csv"
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("row_type", "dataset", "name", "mean", "sample_std")
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--experiment_name", required=True)
    args = parser.parse_args()
    output = summarize(args.experiment_dir, args.experiment_name)
    print(f"Seed summary saved: {output}")


if __name__ == "__main__":
    main()
