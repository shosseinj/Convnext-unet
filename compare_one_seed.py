"""Aggregate completed seed-42 evaluations into a compact comparison CSV."""

import argparse
import csv
import json
from pathlib import Path

from ablation_artifacts import DATASETS, METRICS, validate_evaluation_summary


def compare_experiments(root):
    root = Path(root)
    rows = []
    for experiment_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        summary_path = experiment_dir / "seed_42" / "evaluation_summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"Missing evaluation: {summary_path}")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        validation = validate_evaluation_summary(
            summary_path, payload.get("experiment_name"), 42
        )
        if not validation.valid:
            raise RuntimeError(f"Invalid evaluation {summary_path}: {validation.reason}")
        row = {
            "experiment": experiment_dir.name,
            "seed": 42,
            "trainable_parameters": payload["trainable_parameters"],
            "total_parameters": payload["total_parameters"],
        }
        for metric in METRICS:
            values = [payload["results"][dataset][metric] for dataset in DATASETS]
            row[f"mean_{metric}"] = sum(values) / len(values)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No experiment evaluations found beneath {root}")
    output = root / "comparison.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"One-seed comparison saved: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    compare_experiments(args.root)


if __name__ == "__main__":
    main()
