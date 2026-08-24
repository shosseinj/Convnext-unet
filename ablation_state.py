"""Inspect one ablation seed and emit its training/evaluation state as JSON."""

import argparse
import json
from pathlib import Path

from ablation_artifacts import checkpoint_sha256, validate_evaluation_summary
from ablation_registry import get_experiment
from checkpoint_management import prepare_checkpoint


def inspect_state(
    experiment_name, seed, seed_dir, max_epochs, log_path,
    allow_completed_resume=False,
):
    seed_dir = Path(seed_dir)
    decision = prepare_checkpoint(
        seed_dir, get_experiment(experiment_name), seed, max_epochs, Path(log_path),
        allow_completed_resume=allow_completed_resume,
    )
    evaluation_valid = False
    evaluation_reason = "training is not complete"
    if decision.action == "skip":
        fingerprint = checkpoint_sha256(decision.checkpoint_path)
        validation = validate_evaluation_summary(
            seed_dir / "evaluation_summary.json", experiment_name, seed, fingerprint
        )
        evaluation_valid = validation.valid
        evaluation_reason = validation.reason
    return {
        "training_action": decision.action,
        "training_reason": decision.reason,
        "checkpoint_path": str(decision.checkpoint_path) if decision.checkpoint_path else None,
        "evaluation_valid": evaluation_valid,
        "evaluation_reason": evaluation_reason,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seed_dir", type=Path, required=True)
    parser.add_argument("--max_epochs", type=int, required=True)
    parser.add_argument("--log_path", type=Path, required=True)
    parser.add_argument("--allow_completed_resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(inspect_state(
        args.experiment_name, args.seed, args.seed_dir, args.max_epochs, args.log_path,
        allow_completed_resume=args.allow_completed_resume,
    )))


if __name__ == "__main__":
    main()
