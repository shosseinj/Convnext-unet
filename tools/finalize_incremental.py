#!/usr/bin/env python3
"""Validate, evaluate and aggregate all completed incremental official runs."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    from .run_official_queue import validate_completed_run
except ImportError:
    from run_official_queue import validate_completed_run


def identities(matrix_path):
    matrix = yaml.safe_load(Path(matrix_path).read_text(encoding="utf-8"))
    return [(row["id"], int(seed)) for row in matrix["incremental"] for seed in matrix["seeds"]]


def require_all_complete(root, jobs):
    failures = []
    for variant, seed in jobs:
        path = root / "results" / "raw" / variant / f"seed_{seed}" / "official"
        valid, reason = validate_completed_run(path, variant, seed)
        if not valid:
            failures.append({"variant": variant, "seed": seed, "reason": reason})
    if failures:
        raise SystemExit(f"Incremental finalization blocked: {len(failures)} run(s) incomplete; first={failures[0]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    jobs = identities(root / "configs" / "ablation_matrix.yaml")
    require_all_complete(root, jobs)
    if args.check_only:
        print("Incremental completion gate PASS: 18/18")
        return

    events = []
    for variant, seed in jobs:
        commands = [
            [args.python, str(root / "tools" / "validate_official_run.py"),
             "--variant", variant, "--seed", str(seed)],
            [args.python, str(root / "tools" / "evaluate_official.py"),
             "--variant", variant, "--seed", str(seed), "--device", args.device],
        ]
        for command in commands:
            subprocess.run(command, cwd=root, check=True)
            events.append({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                           "variant": variant, "seed": seed, "command": command, "status": "PASS"})
    for variant in dict.fromkeys(variant for variant, _ in jobs):
        command = [args.python, str(root / "tools" / "aggregate_official.py"), "--variant", variant]
        subprocess.run(command, cwd=root, check=True)
        events.append({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                       "variant": variant, "command": command, "status": "PASS"})
    ledger = root / "results" / "aggregated" / "official" / "finalization_ledger.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps({"status": "PASS", "events": events}, indent=2), encoding="utf-8")
    print(f"Incremental finalization PASS: {ledger}")


if __name__ == "__main__":
    main()
