#!/usr/bin/env python3
"""Run independent special controls only after all 18 incremental runs pass."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    from .run_official_queue import acquire_lock, validate_completed_run
except ImportError:  # Direct script execution places tools/ on sys.path.
    from run_official_queue import acquire_lock, validate_completed_run


def load_control_jobs(matrix_path):
    matrix = yaml.safe_load(Path(matrix_path).read_text(encoding="utf-8"))
    seeds = tuple(matrix.get("seeds", ()))
    controls = tuple(row["id"] for row in matrix.get("controls", ()))
    if len(seeds) != 3 or not controls:
        raise ValueError("Control queue requires three seeds and at least one independent control")
    if len(controls) != len(set(controls)):
        raise ValueError("Duplicate independent control id")
    return tuple((variant, seed) for variant in controls for seed in seeds)


def require_incremental_gate(root, matrix):
    failures = []
    for row in matrix.get("incremental", ()):
        for seed in matrix.get("seeds", ()):
            variant = row["id"]
            run_dir = root / "results" / "raw" / variant / f"seed_{seed}" / "official"
            valid, reason = validate_completed_run(run_dir, variant, seed)
            if not valid:
                failures.append(f"{variant}/seed_{seed}: {reason}")
    if failures:
        raise SystemExit("Control queue blocked until the 18-run incremental gate passes: " + failures[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    matrix_path = root / "configs" / "ablation_matrix.yaml"
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    require_incremental_gate(root, matrix)

    queue_log = root / "results" / "raw" / "control_queue.jsonl"
    lock_path = root / ".agentic" / "control_queue.lock.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    owned_lock = acquire_lock(lock_path)
    try:
        for variant, seed in load_control_jobs(matrix_path):
            run_dir = root / "results" / "raw" / variant / f"seed_{seed}" / "official"
            valid, reason = validate_completed_run(run_dir, variant, seed)
            if valid:
                continue
            command = [args.python, str(root / "train_research.py"), "--variant", variant,
                       "--seed", str(seed), "--device", args.device, "--resume"]
            event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": "start",
                     "variant": variant, "seed": seed, "command": command,
                     "resume_reason": reason, "queue_pid": owned_lock["pid"]}
            with queue_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            result = subprocess.run(command, cwd=root)
            valid_after, validation_reason = validate_completed_run(run_dir, variant, seed)
            event.update({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": "finish",
                          "returncode": result.returncode, "artifact_valid": valid_after,
                          "validation": validation_reason})
            with queue_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            if result.returncode or not valid_after:
                raise SystemExit(f"Control queue stopped: {variant} seed={seed}; {validation_reason}")
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
        if current == owned_lock:
            lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
