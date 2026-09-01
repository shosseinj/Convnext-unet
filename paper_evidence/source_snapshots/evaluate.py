"""Evaluate one completed ablation seed from its best validation checkpoint."""

import argparse
from pathlib import Path

import torch

from ablation_artifacts import DATASETS, atomic_write_json, checkpoint_sha256
from ablation_registry import get_experiment
from checkpoint_management import load_checkpoint_file, strip_thop_state
from evaluation_core import count_parameters, evaluate_loader, make_loader, measure_complexity
from models.convnext_pretrain import ConvNeXtUNet


def build_model(config, encoder_weights, device):
    if config.name.startswith("one_seed_"):
        from one_seed_models import build_experiment_model
        return build_experiment_model(config, encoder_weights, device)
    return ConvNeXtUNet(
        weights_path=encoder_weights,
        num_classes=1,
        encoder_depth=[3, 3, 9, 3],
        enable_msc=config.enable_msc,
        skip_mode=config.skip_mode,
        detail_channels=config.detail_channels,
        enable_gdf=config.enable_gdf,
        detail_fusion_mode=config.detail_fusion_mode,
        deep_supervision_heads=config.deep_supervision_heads,
    ).to(device)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seed_dir", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, default=Path("data"))
    parser.add_argument("--encoder_weights", type=Path, default=Path("convnext_tiny_22k_1k_384.pth"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0, help="Verification-only bounded evaluation")
    parser.add_argument("--tta", action="store_true", help="Use the defined five-view TTA protocol")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_experiment(args.experiment_name)
    checkpoint_path = args.seed_dir / "best_checkpoint.pth"
    checkpoint = load_checkpoint_file(checkpoint_path)
    if checkpoint.get("training_complete") is not True:
        raise RuntimeError(f"Checkpoint training is incomplete: {checkpoint_path}")
    if checkpoint.get("experiment_name") != config.name or checkpoint.get("seed") != args.seed:
        raise RuntimeError("Checkpoint experiment/seed metadata does not match evaluation request")
    if checkpoint.get("architecture") != config.to_dict():
        raise RuntimeError("Checkpoint architecture metadata does not match evaluation request")

    device = torch.device(args.device)
    model = build_model(config, args.encoder_weights, device)
    state = strip_thop_state(checkpoint["model_state_dict"])
    model.load_state_dict(state, strict=True)
    trainable, total = count_parameters(model)
    complexity = measure_complexity(model)
    results = {}
    for dataset in DATASETS:
        loader, count = make_loader(args.data_path, dataset, args.batch_size, args.num_workers)
        metrics = evaluate_loader(model, loader, device, threshold=0.45,
                                  max_batches=args.max_batches, use_tta=args.tta)
        metrics["samples"] = min(count, args.batch_size * args.max_batches) if args.max_batches else count
        results[dataset] = metrics
        print(f"[{dataset}] mDice={metrics['mDice']:.4f} mIoU={metrics['mIoU']:.4f}", flush=True)
    payload = {
        "experiment_name": config.name,
        "seed": args.seed,
        "tta": bool(args.tta),
        "threshold": 0.45,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_fingerprint": checkpoint_sha256(checkpoint_path),
        "trainable_parameters": trainable,
        "total_parameters": total,
        **complexity,
        "results": results,
    }
    output = args.output or args.seed_dir / "evaluation_summary.json"
    atomic_write_json(output, payload)
    print(f"Evaluation summary saved: {output}", flush=True)


if __name__ == "__main__":
    main()
