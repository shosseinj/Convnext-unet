#!/usr/bin/env python3
"""Protocol-driven training runner for reproducible ConvNeXt-UNet experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from models.architecture_factory import build_variant, load_variant_configs
from research_pipeline.data import ManifestSegmentationDataset
from research_pipeline.losses import DiceBCEBoundaryLoss, supervised_loss
from research_pipeline.reproducibility import seed_everything, seed_worker


ROOT = Path(__file__).resolve().parent
DEVELOPMENT_DATASETS = ("Kvasir-SEG", "CVC-ClinicDB")
BACKBONE_WEIGHT_FILES = {
    "resnet34": "resnet34-b627a593.pth",
    "efficientnet_b0": "efficientnet_b0_rwightman-7f5810bc.pth",
}
EVIDENCE_SOURCE_FILES = (
    "train_research.py",
    "models/architecture_factory.py",
    "models/convnext_pretrain.py",
    "research_pipeline/data.py",
    "research_pipeline/losses.py",
    "research_pipeline/reproducibility.py",
    "configs/training_protocol.yaml",
    "configs/ablation_matrix.yaml",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_metadata(model, device, manifest_path, encoder_weights):
    packages = {}
    for package in ("numpy", "opencv-python", "PyYAML", "timm", "torch", "torchvision"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    source_files = list(EVIDENCE_SOURCE_FILES) + [Path(manifest_path).relative_to(ROOT).as_posix()]
    fingerprints = {
        relative: {"sha256": sha256_file(ROOT / relative), "bytes": (ROOT / relative).stat().st_size}
        for relative in source_files
    }
    cuda_device = None
    if device.type == "cuda":
        cuda_device = {"name": torch.cuda.get_device_name(device),
                       "capability": list(torch.cuda.get_device_capability(device))}
    return {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "cuda_device": cuda_device,
        },
        "model": {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters_at_initialization": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "complexity_evidence": "reports/architecture_gate.json",
            "complexity_convention": "GFLOPs = 2 * MACs at 1x3x352x352",
        },
        "source_fingerprints": fingerprints,
        "pretrained_weights": {
            "path": str(Path(encoder_weights).resolve()),
            "sha256": sha256_file(encoder_weights),
            "bytes": Path(encoder_weights).stat().st_size,
        },
    }


def capture_rng_state(train_generator):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_generator": train_generator.get_state(),
    }


def restore_rng_state(state, train_generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint requires CUDA RNG restoration but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    train_generator.set_state(state["train_generator"])


def validate_resume_metadata(saved, current):
    required = ("variant", "seed", "run_mode", "manifest_sha256", "protocol",
                "source_fingerprints", "pretrained_weights")
    missing = [key for key in required if key not in saved]
    if missing:
        raise RuntimeError(f"Resume checkpoint predates reproducible recovery contract; missing: {missing}")
    mismatched = [key for key in required if saved[key] != current[key]]
    if mismatched:
        raise RuntimeError(f"Resume checkpoint immutable metadata mismatch: {mismatched}")


def update_early_stopping(selection, best_score, epochs_without_improvement, encoder_frozen):
    if selection > best_score:
        return selection, 0, True
    if encoder_frozen:
        return best_score, epochs_without_improvement, False
    return best_score, epochs_without_improvement + 1, False


def limited(loader, maximum):
    for index, batch in enumerate(loader):
        if maximum and index >= maximum:
            break
        yield batch


def binary_scores(probability, target, threshold=0.5):
    pred, truth = probability >= threshold, target >= 0.5
    dims = (1, 2, 3)
    tp = (pred & truth).sum(dims).float(); fp = (pred & ~truth).sum(dims).float()
    fn = (~pred & truth).sum(dims).float()
    dice_den = 2 * tp + fp + fn; iou_den = tp + fp + fn
    dice = torch.where(dice_den > 0, 2 * tp / dice_den, torch.ones_like(tp))
    iou = torch.where(iou_den > 0, tp / iou_den, torch.ones_like(tp))
    return dice, iou


def evaluate(model, loader, device, maximum=0):
    model.eval(); dice_values, iou_values, losses, soft_dice_values, positive_fractions = [], [], [], [], []
    with torch.inference_mode():
        for images, masks, _ in limited(loader, maximum):
            images, masks = images.to(device), masks.to(device)
            output = model(images); output = output[0] if isinstance(output, (tuple, list)) else output
            probability = torch.sigmoid(output)
            dice, iou = binary_scores(probability, masks)
            dice_values.extend(dice.cpu().tolist()); iou_values.extend(iou.cpu().tolist())
            losses.extend(torch.abs(probability - masks).mean((1, 2, 3)).cpu().tolist())
            dims = (1, 2, 3)
            soft = (2 * (probability * masks).sum(dims) + 1.0) / (probability.sum(dims) + masks.sum(dims) + 1.0)
            soft_dice_values.extend(soft.cpu().tolist())
            positive_fractions.extend((probability >= 0.5).float().mean(dims).cpu().tolist())
    return {"dice": float(np.mean(dice_values)), "iou": float(np.mean(iou_values)),
            "soft_dice": float(np.mean(soft_dice_values)),
            "predicted_positive_fraction": float(np.mean(positive_fractions)),
            "mae": float(np.mean(losses)), "samples": len(dice_values)}


def make_loaders(protocol, manifest, data_root, seed):
    size = protocol["input_size"][0]; training = protocol["training"]
    train_sets, validation_loaders = [], {}
    for name in DEVELOPMENT_DATASETS:
        split = manifest["datasets"][name]
        train_sets.append(ManifestSegmentationDataset(data_root, name, split["train"], size, True,
                                                       training["augmentation"]))
        dataset = ManifestSegmentationDataset(data_root, name, split["validation"], size, False)
        validation_loaders[name] = DataLoader(dataset, batch_size=training["batch_size"], shuffle=False,
                                               num_workers=training["num_workers"], worker_init_fn=seed_worker)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=training["batch_size"], shuffle=True,
                              num_workers=training["num_workers"], worker_init_fn=seed_worker, generator=generator)
    return train_loader, validation_loaders, generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(42, 3407, 2026))
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs" / "training_protocol.yaml")
    parser.add_argument("--matrix", type=Path, default=ROOT / "configs" / "ablation_matrix.yaml")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--encoder-weights", type=Path, default=ROOT / "convnext_tiny_22k_1k_384.pth")
    parser.add_argument("--pretrained-root", type=Path, default=ROOT / "pretrained")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "raw")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, help="Smoke override; official runs use protocol maximum")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if args.epochs is not None and not args.smoke:
        raise SystemExit("--epochs override is allowed only with --smoke")
    if args.pilot and args.seed != protocol["pilot"]["seed"]:
        raise SystemExit(f"Pilot requires seed {protocol['pilot']['seed']}")
    if args.pilot:
        epochs = protocol["pilot"]["epochs"]
        args.max_train_batches = protocol["pilot"]["max_train_batches_per_epoch"]
        args.max_val_batches = protocol["pilot"]["max_validation_batches"]
    else:
        epochs = args.epochs or protocol["training"]["max_epochs"]
    variants = load_variant_configs(args.matrix, sections=("incremental", "controls"))
    if args.variant not in variants:
        raise SystemExit(f"Unknown variant: {args.variant}")
    variant_config = variants[args.variant]
    encoder_weights = (args.encoder_weights if variant_config.backbone == "convnext_tiny" else
                       args.pretrained_root / BACKBONE_WEIGHT_FILES[variant_config.backbone])
    if not encoder_weights.is_file():
        raise SystemExit(
            f"Required local pretrained encoder weights missing for {variant_config.backbone}: "
            f"{encoder_weights}"
        )
    manifest_path = ROOT / "configs" / "splits" / f"development_seed_{args.seed}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed_everything(args.seed)
    device = torch.device(args.device)
    model = build_variant(variant_config, weights_path=encoder_weights).to(device)
    train_loader, validation_loaders, train_generator = make_loaders(
        protocol, manifest, args.data_root, args.seed
    )
    cfg = protocol["training"]; loss_cfg = dict(cfg["loss"]); ds_weights = loss_cfg.pop("deep_supervision_weights")
    loss_cfg.pop("name")
    criterion = DiceBCEBoundaryLoss(**loss_cfg)
    encoder_params = list(model.encoder.parameters())
    decoder_params = [parameter for name, parameter in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": encoder_params, "lr": cfg["encoder_lr"]},
                                  {"params": decoder_params, "lr": cfg["decoder_lr"]}],
                                 weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - cfg["freeze_encoder_epochs"]))
    run_mode = "smoke" if args.smoke else ("pilot" if args.pilot else "official")
    run_dir = args.output_root / args.variant / f"seed_{args.seed}" / run_mode
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_commit = "unknown"
    metadata = {"variant": args.variant, "seed": args.seed, "run_mode": run_mode,
                "manifest": str(manifest_path), "manifest_sha256": manifest["sha256"],
                "git_commit": git_commit, "protocol": protocol, "epochs": epochs,
                "max_train_batches": args.max_train_batches, "max_val_batches": args.max_val_batches}
    metadata.update(evidence_metadata(model, device, manifest_path, encoder_weights))
    (run_dir / "resolved_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    history, best_score, epochs_without_improvement, start_epoch = [], -1.0, 0, 0
    elapsed_seconds_before_resume = 0.0
    last_checkpoint = run_dir / "last.pth"
    if args.resume and last_checkpoint.is_file():
        saved = torch.load(last_checkpoint, map_location=device, weights_only=False)
        validate_resume_metadata(saved["metadata"], metadata)
        if "rng_state" not in saved:
            raise RuntimeError("Resume checkpoint predates reproducible RNG recovery contract")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        history = saved["history"]
        best_score = saved["best_score"]
        epochs_without_improvement = saved["epochs_without_improvement"]
        elapsed_seconds_before_resume = float(saved.get("elapsed_seconds_total", 0.0))
        start_epoch = saved["epoch"] + 1
        restore_rng_state(saved["rng_state"], train_generator)
        print(f"Resuming {args.variant} seed {args.seed} from epoch {start_epoch + 1}", flush=True)
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        encoder_frozen = epoch < cfg["freeze_encoder_epochs"]
        for parameter in model.encoder.parameters(): parameter.requires_grad = not encoder_frozen
        model.train(); train_losses = []
        for images, masks, _ in limited(train_loader, args.max_train_batches):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = supervised_loss(model(images), masks, criterion, ds_weights)
            if not torch.isfinite(loss): raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}")
            loss.backward(); optimizer.step(); train_losses.append(float(loss.detach().cpu()))
        metrics = {name: evaluate(model, loader, device, args.max_val_batches)
                   for name, loader in validation_loaders.items()}
        selection = float(np.mean([metrics[name]["dice"] for name in DEVELOPMENT_DATASETS]))
        row = {"epoch": epoch + 1, "encoder_frozen": encoder_frozen,
               "train_loss": float(np.mean(train_losses)), "selection_dice": selection, "validation": metrics}
        history.append(row); print(json.dumps(row), flush=True)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        best_score, epochs_without_improvement, is_best = update_early_stopping(
            selection, best_score, epochs_without_improvement, encoder_frozen
        )
        if is_best:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "selection_dice": selection,
                        "validation": metrics, "metadata": metadata}, run_dir / "best.pth")
        if not encoder_frozen: scheduler.step()
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                    "history": history, "best_score": best_score,
                    "epochs_without_improvement": epochs_without_improvement, "metadata": metadata,
                    "rng_state": capture_rng_state(train_generator),
                    "elapsed_seconds_total": elapsed_seconds_before_resume + time.perf_counter() - started},
                   last_checkpoint)
        if epochs_without_improvement >= cfg["early_stopping_patience"]: break
    summary = {"status": "PASS", "variant": args.variant, "seed": args.seed, "run_mode": run_mode,
               "best_selection_dice": best_score, "epochs_completed": len(history),
               "elapsed_seconds": elapsed_seconds_before_resume + time.perf_counter() - started,
               "run_dir": str(run_dir),
               "completed": len(history) >= epochs or epochs_without_improvement >= cfg["early_stopping_patience"]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
