#!/usr/bin/env python3
"""Deep validation for a completed official run; writes a reproducibility report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

try:
    from run_official_queue import validate_completed_run
except ModuleNotFoundError:
    from tools.run_official_queue import validate_completed_run


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Kvasir-SEG", "CVC-ClinicDB")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_deep(run_dir, variant, seed):
    valid, reason = validate_completed_run(run_dir, variant, seed)
    if not valid:
        raise ValueError(reason)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256") != config.get("manifest_sha256"):
        raise ValueError("Resolved configuration manifest hash mismatch")
    expected_samples = {
        name: manifest["datasets"][name]["counts"]["validation"] for name in DATASETS
    }
    for row in history:
        validation = row.get("validation", {})
        if set(validation) != set(DATASETS):
            raise ValueError(f"Validation dataset mismatch at epoch {row.get('epoch')}")
        selection = sum(float(validation[name]["dice"]) for name in DATASETS) / len(DATASETS)
        if not math.isclose(selection, float(row["selection_dice"]), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Selection formula mismatch at epoch {row['epoch']}")
        for name in DATASETS:
            if validation[name].get("samples") != expected_samples[name]:
                raise ValueError(f"Validation sample count mismatch: epoch={row['epoch']} dataset={name}")
            numeric = [value for key, value in validation[name].items() if key != "samples"]
            if not all(math.isfinite(float(value)) for value in numeric):
                raise ValueError(f"Non-finite validation metric: epoch={row['epoch']} dataset={name}")

    best_path, last_path = run_dir / "best.pth", run_dir / "last.pth"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    best_row = max(history, key=lambda row: row["selection_dice"])
    if not math.isclose(float(best["selection_dice"]), float(best_row["selection_dice"]),
                        rel_tol=0, abs_tol=1e-12):
        raise ValueError("Best checkpoint score does not match history maximum")
    if best.get("epoch") + 1 != best_row["epoch"]:
        raise ValueError("Best checkpoint epoch does not match history")
    if last.get("history") != history or last.get("epoch") + 1 != history[-1]["epoch"]:
        raise ValueError("Last checkpoint history/epoch mismatch")
    if not math.isclose(float(last["best_score"]), float(best_row["selection_dice"]),
                        rel_tol=0, abs_tol=1e-12):
        raise ValueError("Last checkpoint best score mismatch")
    if not best.get("model_state_dict") or not last.get("model_state_dict"):
        raise ValueError("Checkpoint model state is empty")
    if best["metadata"].get("variant") != variant or best["metadata"].get("seed") != seed:
        raise ValueError("Best checkpoint identity mismatch")
    if last["metadata"].get("variant") != variant or last["metadata"].get("seed") != seed:
        raise ValueError("Last checkpoint identity mismatch")
    legacy = "rng_state" not in last
    if legacy and not (run_dir / "active_run_source_fingerprint.json").is_file():
        raise ValueError("Legacy checkpoint lacks external active-run fingerprint")
    if not legacy and "elapsed_seconds_total" not in last:
        raise ValueError("Recovery-contract checkpoint lacks cumulative elapsed time")
    if not math.isclose(float(summary["best_selection_dice"]), float(best_row["selection_dice"]),
                        rel_tol=0, abs_tol=1e-12):
        raise ValueError("Summary best score mismatch")
    return {
        "status": "PASS",
        "variant": variant,
        "seed": seed,
        "epochs_completed": len(history),
        "best_epoch": best_row["epoch"],
        "best_selection_dice": best_row["selection_dice"],
        "recovery_contract": "legacy-uninterrupted" if legacy else "rng-restorable",
        "manifest_sha256": manifest["sha256"],
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "history_sha256": sha256_file(run_dir / "history.json"),
        "resolved_config_sha256": sha256_file(run_dir / "resolved_config.json"),
        "validated_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results" / "raw")
    args = parser.parse_args()
    run_dir = args.results_root / args.variant / f"seed_{args.seed}" / "official"
    try:
        report = validate_deep(run_dir, args.variant, args.seed)
    except Exception as error:
        failure = {"status": "FAIL", "variant": args.variant, "seed": args.seed,
                   "error_type": type(error).__name__, "error": str(error),
                   "validated_utc": datetime.now(timezone.utc).isoformat()}
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "validation.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise SystemExit(f"Official run validation failed: {error}")
    (run_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
