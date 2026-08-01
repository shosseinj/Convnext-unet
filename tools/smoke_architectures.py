#!/usr/bin/env python3
"""Gate 2: synthetic one-epoch forward/backward/checkpoint smoke test."""

import argparse
import gc
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.architecture_factory import build_variant, load_variant_configs


def smoke_variant(config, input_size, device):
    torch.manual_seed(42)
    model = build_variant(config, weights_path=None, drop_path_rate=0.0, dropout_rate=0.0).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    losses = []
    for _ in range(2):
        images = torch.rand(1, 3, input_size, input_size, device=device)
        targets = (torch.rand(1, 1, input_size, input_size, device=device) > 0.7).float()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        tensors = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
        loss = sum(F.binary_cross_entropy_with_logits(output, targets) for output in tensors)
        if not torch.isfinite(loss):
            raise AssertionError(f"{config.id}: non-finite loss")
        loss.backward()
        if not any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
            raise AssertionError(f"{config.id}: no finite gradients")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    with tempfile.TemporaryDirectory(prefix="convnext_gate2_") as directory:
        checkpoint_path = Path(directory) / f"{config.id}.pth"
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        restored = build_variant(config, weights_path=None, drop_path_rate=0.0, dropout_rate=0.0).to(device)
        restored.load_state_dict(checkpoint["model_state_dict"], strict=True)

    result = {"variant": config.id, "status": "PASS", "steps": 2,
              "finite_losses": losses, "finite_backward": True,
              "optimizer_step": True, "checkpoint_round_trip": True}
    del model, restored, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=ROOT / "configs" / "ablation_matrix.yaml")
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "smoke_gate.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    results = []
    for config in load_variant_configs(args.matrix).values():
        print(f"Smoke testing {config.id}...", flush=True)
        result = smoke_variant(config, args.input_size, device)
        results.append(result)
        print(f"  PASS losses={result['finite_losses']}", flush=True)
    report = {"gate": "smoke", "status": "PASS", "scope": "synthetic two-batch epoch",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "input_size": args.input_size,
              "device": str(device), "variants": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Gate 2 PASS: {args.output}")


if __name__ == "__main__":
    main()
