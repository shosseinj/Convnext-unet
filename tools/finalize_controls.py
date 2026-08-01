#!/usr/bin/env python3
"""Validate, evaluate and aggregate all independent official control runs."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from .run_control_queue import load_control_jobs
    from .run_official_queue import validate_completed_run
except ImportError:
    from run_control_queue import load_control_jobs
    from run_official_queue import validate_completed_run


ROOT = Path(__file__).resolve().parents[1]


def require_all_complete(root, jobs):
    failures = []
    for variant, seed in jobs:
        run_dir = root / "results" / "raw" / variant / f"seed_{seed}" / "official"
        valid, reason = validate_completed_run(run_dir, variant, seed)
        if not valid:
            failures.append({"variant": variant, "seed": seed, "reason": reason})
    if failures:
        raise SystemExit(f"Control finalization blocked: {len(failures)} run(s) incomplete; first={failures[0]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    jobs = load_control_jobs(ROOT / "configs" / "ablation_matrix.yaml")
    require_all_complete(ROOT, jobs)
    if args.check_only:
        print("Control completion gate PASS: 42/42")
        return
    events = []
    for variant, seed in jobs:
        for command in (
            [args.python, str(ROOT / "tools" / "validate_official_run.py"),
             "--variant", variant, "--seed", str(seed)],
            [args.python, str(ROOT / "tools" / "evaluate_official.py"),
             "--variant", variant, "--seed", str(seed), "--device", args.device],
        ):
            subprocess.run(command, cwd=ROOT, check=True)
            events.append({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                           "variant": variant, "seed": seed, "command": command, "status": "PASS"})
    for variant in dict.fromkeys(variant for variant, _ in jobs):
        command = [args.python, str(ROOT / "tools" / "aggregate_official.py"), "--variant", variant]
        subprocess.run(command, cwd=ROOT, check=True)
        events.append({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                       "variant": variant, "command": command, "status": "PASS"})
    ledger = ROOT / "results" / "aggregated" / "official" / "control_finalization_ledger.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps({"status": "PASS", "events": events}, indent=2), encoding="utf-8")
    print(f"Control finalization PASS: {ledger}")


if __name__ == "__main__":
    main()
