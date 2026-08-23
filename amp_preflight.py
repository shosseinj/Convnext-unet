"""Select a shared AMP batch size using one optimized UGBR train step."""

import argparse
import gc
import json
from pathlib import Path

import torch

from ablation_registry import get_experiment
from amp_training import autocast_context, create_grad_scaler
from one_seed_models import build_experiment_model
from research_pipeline.losses import DiceBCEBoundaryLoss, ugbr_composite_loss


def candidate_batch_sizes(start_batch_size):
    allowed = (24, 20, 16)
    if start_batch_size not in allowed:
        raise ValueError(f"Unsupported starting batch size: {start_batch_size}")
    return allowed[allowed.index(start_batch_size):]


def run_step(batch_size, encoder_weights):
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = build_experiment_model(
        get_experiment("one_seed_02_add_ugbr"), encoder_weights, device
    ).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = create_grad_scaler(True, "cuda")
    criterion = DiceBCEBoundaryLoss(
        dice_weight=0.55, bce_weight=0.25, boundary_weight=0.20
    )
    images = torch.rand(batch_size, 3, 352, 352, device=device)
    targets = (torch.rand(batch_size, 1, 352, 352, device=device) > 0.7).float()
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(True, "cuda"):
        outputs = model(images)
        loss = ugbr_composite_loss(outputs, targets, criterion)["total"]
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.5)
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    result = {
        "batch_size": batch_size,
        "loss": float(loss.detach()),
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
    }
    del targets, images, loss, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def select_batch_size(start_batch_size, encoder_weights):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for AMP batch-size preflight")
    failures = []
    for batch_size in candidate_batch_sizes(start_batch_size):
        try:
            result = run_step(batch_size, encoder_weights)
            result["failures"] = failures
            return result
        except torch.OutOfMemoryError as exc:
            failures.append({"batch_size": batch_size, "reason": str(exc)})
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError(f"AMP preflight failed at every allowed batch size: {failures}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-batch-size", type=int, default=24)
    parser.add_argument(
        "--encoder-weights", type=Path,
        default=Path("convnext_tiny_22k_1k_384.pth"),
    )
    args = parser.parse_args()
    print(json.dumps(select_batch_size(args.start_batch_size, args.encoder_weights)))


if __name__ == "__main__":
    main()
