#!/usr/bin/env python3
"""Run the fixed seed-42 UGBR pilot queue sequentially and fail closed."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PILOT_JOBS = (
    "baseline",
    "baseline_ugbr",
    "baseline_best_existing",
    "baseline_best_existing_ugbr",
)
SEED = 42
PILOT_NAMESPACE = Path("results/pilot_campaigns/ugbr_seed42_v2")
EVIDENCE_SOURCE_FILES = (
    "train_research.py",
    "models/architecture_factory.py",
    "models/convnext_pretrain.py",
    "models/ugbr.py",
    "research_pipeline/data.py",
    "research_pipeline/losses.py",
    "research_pipeline/reproducibility.py",
    "configs/training_protocol.yaml",
    "configs/ablation_matrix.yaml",
    "configs/splits/development_seed_42.json",
)
STATUS_FIELDS = (
    "Stage", "Workflow status", "Active agent", "Active run", "Latest epoch",
    "Completed pilots", "ConvNeXt audit", "Last validation", "Process status",
    "Last error", "Next action", "Console log", "Last update",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_alive(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
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
    """Atomically acquire a PID lock, removing it only after its PID is stale."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            if process_alive(existing.get("pid")):
                raise SystemExit(f"Pilot campaign already active with PID {existing['pid']}: {path}")
            path.unlink(missing_ok=True)
            continue
        payload = {"pid": os.getpid(), "created_utc": utc_now(), "queue": list(PILOT_JOBS)}
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return payload
    raise SystemExit(f"Could not safely acquire pilot campaign lock: {path}")


def emit(path, event, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp_utc": utc_now(), "event": event, **fields}) + "\n")


def write_status(root, *, workflow_status, active_run, completed, validation,
                 process_status, error, next_action):
    values = {
        "Stage": "PILOT", "Workflow status": workflow_status,
        "Active agent": "Experiment Agent", "Active run": active_run,
        "Latest epoch": "See active run history.json" if active_run != "None" else "None",
        "Completed pilots": f"{completed}/{len(PILOT_JOBS)}",
        "ConvNeXt audit": "PASS", "Last validation": validation,
        "Process status": process_status, "Last error": error,
        "Next action": next_action, "Console log": "CAMPAIGN_CONSOLE.log",
        "Last update": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    lines = [f"{field}: {values[field]}" for field in STATUS_FIELDS]
    (Path(root) / "RUN_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def expected_metadata(root, variant):
    root = Path(root)
    protocol = yaml.safe_load((root / "configs/training_protocol.yaml").read_text(encoding="utf-8"))
    matrix = yaml.safe_load((root / "configs/ablation_matrix.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((root / "configs/splits/development_seed_42.json").read_text(encoding="utf-8"))
    ids = [row["id"] for row in matrix.get("incremental", ())] + [
        row["id"] for row in matrix.get("pilot", ())
    ]
    if tuple(dict.fromkeys(name for name in PILOT_JOBS if name in ids)) != PILOT_JOBS:
        raise ValueError("Pilot matrix does not define the exact required four variants")
    fingerprints = {
        relative: {"sha256": sha256_file(root / relative), "bytes": (root / relative).stat().st_size}
        for relative in EVIDENCE_SOURCE_FILES
    }
    return {
        "variant": variant, "seed": SEED, "run_mode": "pilot",
        "manifest_sha256": manifest["sha256"], "protocol": protocol,
        "source_fingerprints": fingerprints,
        "epochs": protocol["pilot"]["epochs"],
        "max_train_batches": protocol["pilot"]["max_train_batches_per_epoch"],
        "max_val_batches": protocol["pilot"]["max_validation_batches"],
    }


def metadata_equivalent(config, expected):
    keys = tuple(expected)
    mismatched = [key for key in keys if config.get(key) != expected[key]]
    return not mismatched, mismatched


def pilot_run_dir(root, variant):
    """Return the isolated fresh-pilot directory; legacy raw results are never targeted."""
    return Path(root) / PILOT_NAMESPACE / variant / "seed_42" / "pilot"


def validate_run(root, variant):
    """Strict exact-protocol validation suitable for reuse and post-run gating."""
    run_dir = pilot_run_dir(root, variant)
    required = ("summary.json", "resolved_config.json", "history.json", "best.pth", "last.pth")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        return False, f"missing artifacts: {missing}"
    try:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        expected = expected_metadata(root, variant)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, yaml.YAMLError) as error:
        return False, f"unreadable validation evidence: {error}"
    equivalent, mismatched = metadata_equivalent(config, expected)
    if not equivalent:
        return False, f"protocol fingerprint mismatch: {mismatched}"
    if summary.get("status") != "PASS" or summary.get("completed") is not True:
        return False, "summary is not completed PASS"
    if (summary.get("variant"), summary.get("seed"), summary.get("run_mode")) != (variant, SEED, "pilot"):
        return False, "summary identity mismatch"
    if not history or len(history) != summary.get("epochs_completed"):
        return False, "history length mismatch"
    try:
        finite = all(math.isfinite(float(row[key])) for row in history
                     for key in ("train_loss", "selection_dice"))
    except (KeyError, TypeError, ValueError):
        finite = False
    if not finite:
        return False, "history contains missing or non-finite metrics"
    if min((run_dir / name).stat().st_size for name in ("best.pth", "last.pth")) <= 0:
        return False, "checkpoint is empty"
    return True, "exact-protocol artifact validation PASS"


def resume_compatible(root, variant):
    run_dir = pilot_run_dir(root, variant)
    checkpoint = run_dir / "last.pth"
    config_path = run_dir / "resolved_config.json"
    if not checkpoint.is_file():
        existing = run_dir.exists() and any(run_dir.iterdir())
        return (False, "incomplete artifacts without resumable checkpoint") if existing else (False, "fresh run")
    if not config_path.is_file():
        return False, "checkpoint exists without resolved configuration"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        equivalent, mismatched = metadata_equivalent(config, expected_metadata(root, variant))
    except Exception as error:
        return False, f"resume metadata unreadable: {error}"
    return (True, "compatible checkpoint") if equivalent else (False, f"resume mismatch: {mismatched}")


def training_command(python, root, variant, device, resume=False):
    command = [python, "-u", str(Path(root) / "train_research.py"), "--variant", variant,
               "--seed", str(SEED), "--device", device, "--pilot",
               "--output-root", str(Path(root) / PILOT_NAMESPACE)]
    if resume:
        command.append("--resume")
    return command


def validation_command(python, script, root, variant):
    return [python, "-u", str(script), "--validate-run", variant, "--root", str(root)]


def run_campaign(root, python, device, run=subprocess.run):
    root = Path(root).resolve()
    ledger = root / PILOT_NAMESPACE / "ugbr_pilot_queue.jsonl"
    lock_path = root / ".agentic/ugbr_pilot_campaign.lock.json"
    owned_lock = acquire_lock(lock_path)
    completed = 0
    emit(ledger, "campaign_start", pid=owned_lock["pid"], queue=list(PILOT_JOBS), seed=SEED)
    try:
        for variant in PILOT_JOBS:
            valid, reason = validate_run(root, variant)
            if valid:
                completed += 1
                emit(ledger, "reuse", variant=variant, seed=SEED, evidence=reason)
                continue
            can_resume, resume_reason = resume_compatible(root, variant)
            run_dir = pilot_run_dir(root, variant)
            if run_dir.exists() and any(run_dir.iterdir()) and not can_resume:
                raise RuntimeError(f"Preserved incompatible prior run {variant}: {resume_reason}")
            command = training_command(python, root, variant, device, can_resume)
            write_status(root, workflow_status="RUNNING", active_run=variant, completed=completed,
                         validation=reason, process_status="Visible terminal active", error="None",
                         next_action="Train active pilot, then validate artifacts")
            emit(ledger, "run_start", variant=variant, seed=SEED, command=command,
                 resume=can_resume, resume_reason=resume_reason)
            result = run(command, cwd=root)
            if result.returncode:
                raise RuntimeError(f"training failed for {variant}: returncode={result.returncode}")
            check = validation_command(python, Path(__file__).resolve(), root, variant)
            checked = run(check, cwd=root)
            valid, reason = validate_run(root, variant)
            emit(ledger, "run_validation", variant=variant, seed=SEED,
                 validator_returncode=checked.returncode, valid=valid, evidence=reason)
            if checked.returncode or not valid:
                raise RuntimeError(f"validation failed for {variant}: {reason}")
            completed += 1
        write_status(root, workflow_status="PILOT_VALIDATION_PENDING", active_run="None",
                     completed=completed, validation="All per-run artifact gates PASS",
                     process_status="Stopped normally", error="None",
                     next_action="Validation Agent independently review pilot evidence")
        emit(ledger, "campaign_complete", completed=completed, total=len(PILOT_JOBS))
        return 0
    except BaseException as error:
        emit(ledger, "campaign_error", completed=completed, error_type=type(error).__name__, error=str(error))
        write_status(root, workflow_status="FAILED", active_run="None", completed=completed,
                     validation="FAIL", process_status="Stopped", error=str(error),
                     next_action="Inspect CAMPAIGN_CONSOLE.log and queue ledger")
        raise
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
        if current == owned_lock:
            lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validate-run", choices=PILOT_JOBS)
    args = parser.parse_args()
    if args.validate_run:
        valid, reason = validate_run(args.root, args.validate_run)
        print(json.dumps({"status": "PASS" if valid else "FAIL", "variant": args.validate_run,
                          "seed": SEED, "evidence": reason}))
        return 0 if valid else 1
    return run_campaign(args.root, args.python, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
