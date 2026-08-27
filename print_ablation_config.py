"""Print one registered ablation configuration as JSON for runner scripts."""

import argparse
import json
from dataclasses import asdict

import torch

from ablation_registry import get_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = get_experiment(args.experiment)
    payload = asdict(config)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("experiment_name") != args.experiment:
            raise ValueError("Checkpoint experiment metadata does not match")
        if args.seed is not None and checkpoint.get("seed") != args.seed:
            raise ValueError("Checkpoint seed metadata does not match")
        if checkpoint.get("architecture") != config.to_dict():
            raise ValueError("Checkpoint architecture metadata does not match")
        payload["source_epoch"] = int(checkpoint["epoch"]) + 1
        payload["has_optimizer_state"] = "optimizer_state_dict" in checkpoint
        payload["has_ema_state"] = "ema_state_dict" in checkpoint
        payload["has_scaler_state"] = any(
            key in checkpoint for key in ("scaler_state_dict", "grad_scaler_state_dict")
        )
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
