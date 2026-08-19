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


def deep_validation_pass(run_dir, variant, seed):
    path = run_dir / "validation.json"
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return report.get("status") == "PASS" and report.get("variant") == variant and report.get("seed") == seed


def write_run_status(root, *, campaign_status, active_run, latest_epoch, completed,
                     last_completed, last_result, process_status, last_error, next_action):
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "Stage: EXPERIMENT",
        f"Campaign status: {campaign_status}",
        f"Active run: {active_run}",
        f"Latest epoch: {latest_epoch}",
        f"Completed runs: {completed}/18",
        f"Last completed run: {last_completed}",
        f"Last result: {last_result}",
        f"Process status: {process_status}",
        f"Last error: {last_error}",
        f"Next action: {next_action}",
        f"Console log: {root / 'CAMPAIGN_CONSOLE.log'}",
        f"Last update: {timestamp}",
    ]
    (root / "RUN_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_deep_validation(root, python, run_dir, variant, seed):
    command = [python, str(root / "tools" / "validate_official_run.py"),
               "--variant", variant, "--seed", str(seed),
               "--results-root", str(run_dir.parents[2])]
    result = subprocess.run(command, cwd=root)
    return result.returncode == 0 and deep_validation_pass(run_dir, variant, seed)


def training_command(python, script, variant, seed, device, resume_checkpoint, resume=True):
    arguments = [str(script), "--variant", variant, "--seed", str(seed),
                 "--device", device]
    if not resume:
        return [python, *arguments]
    arguments.append("--resume")
    if not resume_checkpoint.is_file():
        return [python, *arguments]
    wrapper = (
        "import runpy,sys,torch;"
        "_real_load=torch.load;"
        "torch.load=lambda *a,**kw:_real_load(*a,**dict(kw,map_location='cpu'));"
        "_script=sys.argv[1];sys.argv=sys.argv[1:];"
        "runpy.run_path(_script,run_name='__main__')"
    )
    return [python, "-u", "-c", wrapper, *arguments]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-root", type=Path, default=Path("results/raw"))
    parser.add_argument("--only-variant")
    parser.add_argument("--only-seed", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    queue_log = output_root / "official_queue.jsonl"
    queue_log.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".agentic" / "official_queue.lock.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    owned_lock = acquire_lock(lock_path)
    try:
        jobs = load_incremental_jobs(root / "configs" / "ablation_matrix.yaml")
        if args.only_variant is not None or args.only_seed is not None:
            jobs = tuple((variant, seed) for variant, seed in jobs
                         if (args.only_variant is None or variant == args.only_variant)
                         and (args.only_seed is None or seed == args.only_seed))
            if not jobs:
                raise SystemExit("Recovery mapping does not match a canonical official job")
        completed = 0
        for variant, seed in jobs:
                run_dir = output_root / variant / f"seed_{seed}" / "official"
                valid, reason = validate_completed_run(run_dir, variant, seed)
                if valid:
                    if not deep_validation_pass(run_dir, variant, seed):
                        if not run_deep_validation(root, args.python, run_dir, variant, seed):
                            raise SystemExit(f"Official queue stopped: deep validation failed for {variant} seed={seed}")
                    completed += 1
                    print(f"SKIP completed {variant} seed={seed}: {reason}", flush=True)
                    continue
                now = datetime.now(timezone.utc).isoformat()
                print("=" * 64, flush=True)
                print(f"RUN START | variant={variant} | seed={seed} | time={now}", flush=True)
                print("=" * 64, flush=True)
                write_run_status(root, campaign_status="RUNNING", active_run=f"{variant} / seed {seed}",
                                 latest_epoch="Starting", completed=completed,
                                 last_completed="None" if completed == 0 else "See validated artifacts",
                                 last_result="Run started", process_status="Visible terminal active",
                                 last_error="None", next_action="Train and validate the active run")
                command = training_command(
                    args.python, root / "train_research.py", variant, seed, args.device,
                     run_dir / "last.pth", resume=not args.no_resume
                 )
                command.extend(["--output-root", str(output_root)])
                event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": "start",
                         "variant": variant, "seed": seed, "command": command,
                         "python": args.python, "device": args.device, "resume_reason": reason,
                         "queue_pid": os.getpid()}
                with queue_log.open("a", encoding="utf-8") as handle: handle.write(json.dumps(event) + "\n")
                result = subprocess.run(command, cwd=root)
                valid_after, validation_reason = validate_completed_run(run_dir, variant, seed)
                deep_valid = False
                if result.returncode == 0 and valid_after:
                    deep_valid = run_deep_validation(root, args.python, run_dir, variant, seed)
                event.update({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": "finish",
                              "returncode": result.returncode, "artifact_valid": valid_after,
                              "validation": validation_reason, "deep_validation_pass": deep_valid})
                with queue_log.open("a", encoding="utf-8") as handle: handle.write(json.dumps(event) + "\n")
                if result.returncode or not valid_after or not deep_valid:
                    error = (f"returncode={result.returncode}; artifact_valid={valid_after}; "
                             f"deep_validation_pass={deep_valid}; {validation_reason}")
                    print(f"RUN END | variant={variant} | seed={seed} | status=FAIL | "
                          f"time={datetime.now(timezone.utc).isoformat()}", flush=True)
                    write_run_status(root, campaign_status="FAILED", active_run="None",
                                     latest_epoch="See run history", completed=completed,
                                     last_completed="None" if completed == 0 else "See validated artifacts",
                                     last_result="Official validation failed", process_status="Stopped",
                                     last_error=error, next_action="Inspect the console log and run artifacts")
                    raise SystemExit(
                        f"Official queue stopped: {variant} seed={seed} returncode={result.returncode}; "
                        f"validation={validation_reason}; deep_validation_pass={deep_valid}"
                    )
                completed += 1
                print(f"RUN END | variant={variant} | seed={seed} | status=PASS | "
                      f"time={datetime.now(timezone.utc).isoformat()}", flush=True)
                write_run_status(root, campaign_status="RUNNING", active_run="None",
                                 latest_epoch="Complete", completed=completed,
                                 last_completed=f"{variant} / seed {seed}",
                                 last_result="Official validation PASS", process_status="Transitioning",
                                 last_error="None", next_action="Start the next pending official run")
        write_run_status(root, campaign_status="COMPLETE", active_run="None", latest_epoch="None",
                         completed=len(jobs), last_completed=f"{jobs[-1][0]} / seed {jobs[-1][1]}",
                         last_result="Official validation PASS", process_status="Stopped normally",
                         last_error="None",
                         next_action="Review aggregate official results before entering the next stage")
        print(f"CAMPAIGN COMPLETE | runs={len(jobs)}/{len(jobs)} | "
              f"time={datetime.now(timezone.utc).isoformat()}", flush=True)
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
        if current == owned_lock:
            lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
