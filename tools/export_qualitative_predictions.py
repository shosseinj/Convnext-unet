#!/usr/bin/env python3
"""Export deterministic qualitative masks from validated ablation checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ablation_registry import get_experiment
from checkpoint_management import load_checkpoint_file, strip_thop_state
from evaluate import build_model
from evaluation_core import DATASET_DIRECTORIES, _main_logits
from tools.generate_qualitative_ablation_examples import DATASETS


def selected_images(data_root: Path, count: int = 2) -> dict[str, list[Path]]:
    selected: dict[str, list[Path]] = {}
    selected_fingerprints: list[tuple[np.ndarray, np.ndarray]] = []
    for dataset in DATASETS:
        directory = data_root / DATASET_DIRECTORIES[dataset]
        image_by_stem = {
            path.stem.casefold(): path for path in (directory / "images").iterdir()
            if path.is_file() and path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        }
        rows = []
        for mask_path in sorted((directory / "masks").iterdir(), key=lambda path: path.name.casefold()):
            image_path = image_by_stem.get(mask_path.stem.casefold())
            if image_path is None:
                continue
            with Image.open(mask_path) as mask:
                values = np.asarray(mask.convert("L"))
            decoded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if decoded is None:
                continue
            gray = cv2.cvtColor(cv2.resize(decoded, (9, 8)), cv2.COLOR_BGR2GRAY)
            dhash = (gray[:, 1:] > gray[:, :-1]).reshape(-1)
            thumbnail = cv2.resize(decoded, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
            rows.append({"image": image_path.name, "mask_area_ratio": np.count_nonzero(values >= 128) / values.size,
                         "path": image_path, "dhash": dhash, "thumbnail": thumbnail})
        ranked = sorted(rows, key=lambda row: (row["mask_area_ratio"], row["image"].casefold()))
        dataset_selection = []
        for selection_index in range(count):
            target = ((selection_index + 1) / (count + 1)) * (len(ranked) - 1)
            candidates = sorted(enumerate(ranked), key=lambda item: (abs(item[0] - target), item[1]["image"].casefold()))
            chosen = None
            for _, candidate in candidates:
                fingerprint = (candidate["dhash"], candidate["thumbnail"])
                is_near_duplicate = any(
                    np.count_nonzero(fingerprint[0] != prior_hash) <= 22
                    and np.mean((fingerprint[1] - prior_thumbnail) ** 2) < 1000.0
                    for prior_hash, prior_thumbnail in selected_fingerprints
                )
                if not is_near_duplicate and all(candidate["path"] != item["path"] for item in dataset_selection):
                    chosen = candidate
                    break
            if chosen is None:
                raise RuntimeError(f"Unable to find {count} perceptually distinct samples for {dataset}")
            dataset_selection.append(chosen)
            selected_fingerprints.append((chosen["dhash"], chosen["thumbnail"]))
        selected[dataset] = [row["path"] for row in dataset_selection]
    return selected


def load_validated_model(experiment: str, seed_dir: Path, encoder_weights: Path, device: torch.device):
    config = get_experiment(experiment)
    checkpoint_path = seed_dir / "best_checkpoint.pth"
    checkpoint = load_checkpoint_file(checkpoint_path)
    if checkpoint.get("training_complete") is not True:
        raise RuntimeError(f"Incomplete checkpoint: {checkpoint_path}")
    if checkpoint.get("experiment_name") != experiment or checkpoint.get("seed") != 42:
        raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint_path}")
    if checkpoint.get("architecture") != config.to_dict():
        raise RuntimeError(f"Checkpoint architecture mismatch: {checkpoint_path}")
    model = build_model(config, encoder_weights, device)
    model.load_state_dict(strip_thop_state(checkpoint["model_state_dict"]), strict=True)
    return model.eval(), checkpoint_path


def predict(model, image_path: Path, device: torch.device, threshold: float) -> Image.Image:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Unable to decode {image_path}")
    height, width = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (352, 352), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(_main_logits(model(tensor)))
        probability = F.interpolate(probability, size=(height, width), mode="bilinear", align_corners=False)
    mask = (probability[0, 0].cpu().numpy() > threshold).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("predictions/qualitative_ablation"))
    parser.add_argument("--encoder-weights", type=Path, default=Path("convnext_tiny_22k_1k_384.pth"))
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    specs = (
        ("baseline", "one_seed_01_baseline", Path("one_seed_results/ablation/01_baseline/seed_42")),
        ("second_ablation", "one_seed_33_fafem_warmup_cosine",
         Path("one_seed_results/ablation/33_fafem_warmup_cosine/seed_42")),
        ("third_ablation", "one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine",
         Path("one_seed_results/ablation/45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine/seed_42")),
    )
    chosen = selected_images(args.data_root.resolve())
    device = torch.device(args.device)
    provenance = {"threshold": args.threshold, "tta": False, "input_size": [352, 352], "seed": 42, "models": {}}
    for label, experiment, seed_dir in specs:
        print(f"Loading {label}: {experiment}", flush=True)
        model, checkpoint_path = load_validated_model(experiment, seed_dir, args.encoder_weights, device)
        provenance["models"][label] = {"experiment": experiment, "checkpoint": str(checkpoint_path.resolve())}
        for dataset, image_paths in chosen.items():
            output_dir = args.output_root / label / dataset
            output_dir.mkdir(parents=True, exist_ok=True)
            expected_names = {f"{image_path.stem}.png" for image_path in image_paths}
            for stale_path in output_dir.glob("*.png"):
                if stale_path.name not in expected_names:
                    stale_path.unlink()
            for image_path in image_paths:
                output_path = output_dir / f"{image_path.stem}.png"
                predict(model, image_path, device, args.threshold).save(output_path, format="PNG", dpi=(300, 300))
                print(f"[{label}] {dataset}/{output_path.name}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "prediction_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Saved 30 prediction masks under {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
