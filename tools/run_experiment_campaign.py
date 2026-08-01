#!/usr/bin/env python3
"""Guard the full incremental-to-control experiment campaign without GPU overlap."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from .finalize_incremental import identities
    from .run_control_queue import load_control_jobs
    from .run_official_queue import acquire_lock, validate_completed_run
except ImportError:
    from finalize_incremental import identities
    from run_control_queue import load_control_jobs
    from run_official_queue import acquire_lock, validate_completed_run


ROOT = Path(__file__).resolve().parents[1]


def emit(path, event, **fields):
    payload = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def valid_deep_report(run_dir, variant, seed):
    path = run_dir / "validation.json"
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("status") == "PASS" and value.get("variant") == variant and value.get("seed") == seed


def validate_completed_jobs(jobs, python, ledger):
    complete = 0
    for variant, seed in jobs:
        run_dir = ROOT / "results" / "raw" / variant / f"seed_{seed}" / "official"
        valid, _ = validate_completed_run(run_dir, variant, seed)
        if not valid:
            continue
        complete += 1
        if valid_deep_report(run_dir, variant, seed):
            continue
        command = [python, str(ROOT / "tools" / "validate_official_run.py"),
                   "--variant", variant, "--seed", str(seed)]
        emit(ledger, "deep_validation_start", variant=variant, seed=seed, command=command)
        result = subprocess.run(command, cwd=ROOT)
        emit(ledger, "deep_validation_finish", variant=variant, seed=seed,
             returncode=result.returncode, artifact_valid=valid_deep_report(run_dir, variant, seed))
        if result.returncode or not valid_deep_report(run_dir, variant, seed):
            raise SystemExit(f"Campaign stopped: deep validation failed for {variant}/seed_{seed}")
    return complete


def run_stage(command, ledger, event):
    emit(ledger, f"{event}_start", command=command)
    result = subprocess.run(command, cwd=ROOT)
    emit(ledger, f"{event}_finish", command=command, returncode=result.returncode)
    if result.returncode:
        raise SystemExit(f"Campaign stopped: {event} returncode={result.returncode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        raise SystemExit("poll-seconds must be at least 10")
    lock_path = ROOT / ".agentic" / "experiment_campaign.lock.json"
    ledger = ROOT / "results" / "raw" / "experiment_campaign.jsonl"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    owned_lock = acquire_lock(lock_path)
    emit(ledger, "campaign_start", pid=owned_lock["pid"], python=args.python, device=args.device)
    try:
        incremental_jobs = identities(ROOT / "configs" / "ablation_matrix.yaml")
        last_count = -1
        while True:
            count = validate_completed_jobs(incremental_jobs, args.python, ledger)
            if count != last_count:
                emit(ledger, "incremental_progress", completed=count, total=len(incremental_jobs))
                last_count = count
            if count == len(incremental_jobs):
                break
            time.sleep(args.poll_seconds)

        run_stage([args.python, str(ROOT / "tools" / "finalize_incremental.py"),
                   "--device", args.device, "--python", args.python], ledger, "incremental_finalization")
        run_stage([args.python, str(ROOT / "tools" / "generate_incremental_result_artifacts.py")],
                  ledger, "incremental_manuscript_artifacts")
        run_stage([args.python, str(ROOT / "tools" / "generate_qualitative_figures.py"),
                   "--device", args.device], ledger, "qualitative_figures")
        run_stage([args.python, str(ROOT / "tools" / "run_control_queue.py"),
                   "--device", args.device, "--python", args.python], ledger, "control_queue")

        control_jobs = load_control_jobs(ROOT / "configs" / "ablation_matrix.yaml")
        count = validate_completed_jobs(control_jobs, args.python, ledger)
        if count != len(control_jobs):
            raise SystemExit(f"Campaign stopped: only {count}/{len(control_jobs)} control runs validated")
        run_stage([args.python, str(ROOT / "tools" / "finalize_controls.py"),
                   "--device", args.device, "--python", args.python], ledger, "control_finalization")
        run_stage([args.python, str(ROOT / "tools" / "generate_control_result_artifacts.py")],
                  ledger, "control_manuscript_artifacts")
        run_stage([args.python, str(ROOT / "tools" / "audit_manuscript_artifacts.py")],
                  ledger, "manuscript_artifact_audit")
        emit(ledger, "campaign_complete", incremental_runs=len(incremental_jobs), control_runs=len(control_jobs))
    except BaseException as error:
        emit(ledger, "campaign_error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
        if current == owned_lock:
            lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
