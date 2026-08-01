#!/usr/bin/env python3
"""Run or resume the 18 official experiments sequentially on one GPU."""

import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


VARIANTS = ("baseline", "baseline_msc", "baseline_msc_bsei", "baseline_msc_bsei_detail",
            "baseline_msc_bsei_detail_gdf", "full")
SEEDS = (42, 3407, 2026)


def load_incremental_jobs(matrix_path):
    matrix = yaml.safe_load(Path(matrix_path).read_text(encoding="utf-8"))
    seeds = tuple(matrix.get("seeds", ()))
    variants = tuple(row["id"] for row in matrix.get("incremental", ()))
    if seeds != SEEDS:
        raise ValueError(f"Canonical seed matrix mismatch: {seeds}")
    if variants != VARIANTS:
        raise ValueError(f"Canonical incremental matrix mismatch: {variants}")
    return tuple((variant, seed) for variant in variants for seed in seeds)


def process_alive(pid):
    try:
        pid = int(pid)
    except (ValueError, TypeError):
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, pid
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_lock(path):
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if process_alive(existing.get("pid")):
            raise SystemExit(f"Official queue already active with PID {existing['pid']}: {path}")
        path.unlink(missing_ok=True)
    payload = {"pid": os.getpid(), "created_utc": datetime.now(timezone.utc).isoformat()}
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return payload


def validate_completed_run(run_dir, variant, seed):
    required = ("summary.json", "resolved_config.json", "history.json", "best.pth", "last.pth")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        return False, f"missing artifacts: {missing}"
    try:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"invalid JSON: {error}"
    if summary.get("status") != "PASS" or summary.get("completed") is not True:
        return False, "summary is not completed PASS"
    if summary.get("variant") != variant or summary.get("seed") != seed:
        return False, "summary experiment identity mismatch"
    if config.get("variant") != variant or config.get("seed") != seed or config.get("run_mode") != "official":
        return False, "resolved configuration identity mismatch"
    if len(history) != summary.get("epochs_completed") or not history:
        return False, "history length does not match summary"
    numeric_fields = ("train_loss", "selection_dice")
    if any(not math.isfinite(float(row[field])) for row in history for field in numeric_fields):
        return False, "history contains non-finite values"
    extended = all(key in config for key in ("environment", "model", "source_fingerprints", "pretrained_weights"))
    legacy_fingerprint = (run_dir / "active_run_source_fingerprint.json").is_file()
    if not extended and not legacy_fingerprint:
        return False, "evidence metadata is incomplete"
    if min((run_dir / name).stat().st_size for name in ("best.pth", "last.pth")) <= 0:
        return False, "checkpoint is empty"
    return True, "validated completed run"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    queue_log = root / "results" / "raw" / "official_queue.jsonl"
    queue_log.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".agentic" / "official_queue.lock.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    owned_lock = acquire_lock(lock_path)
    try:
        for variant, seed in load_incremental_jobs(root / "configs" / "ablation_matrix.yaml"):
                run_dir = root / "results" / "raw" / variant / f"seed_{seed}" / "official"
                valid, reason = validate_completed_run(run_dir, variant, seed)
                if valid:
                    print(f"SKIP completed {variant} seed={seed}: {reason}", flush=True)
                    continue
                command = [args.python, str(root / "train_research.py"), "--variant", variant,
                           "--seed", str(seed), "--device", args.device, "--resume"]
                event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": "start",
                         "variant": variant, "seed": seed, "command": command,
                         "python": args.python, "device": args.device, "resume_reason": reason,
                         "queue_pid": os.getpid()}
                with queue_log.open("a", encoding="utf-8") as handle: handle.write(json.dumps(event) + "\n")
                result = subprocess.run(command, cwd=root)
                valid_after, validation_reason = validate_completed_run(run_dir, variant, seed)
                event.update({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": "finish",
                              "returncode": result.returncode, "artifact_valid": valid_after,
                              "validation": validation_reason})
                with queue_log.open("a", encoding="utf-8") as handle: handle.write(json.dumps(event) + "\n")
                if result.returncode or not valid_after:
                    raise SystemExit(
                        f"Official queue stopped: {variant} seed={seed} returncode={result.returncode}; "
                        f"validation={validation_reason}"
                    )
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
        if current == owned_lock:
            lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
