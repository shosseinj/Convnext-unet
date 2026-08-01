"""Seed and split-manifest utilities."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def list_pairs(dataset_dir: Path) -> list[dict[str, str]]:
    image_dir, mask_dir = dataset_dir / "images", dataset_dir / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and masks/ below {dataset_dir}")
    images = sorted((p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
                    key=lambda path: path.name.lower())
    mask_by_name = {p.name.lower(): p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    pairs, missing = [], []
    for image in images:
        mask = mask_by_name.get(image.name.lower())
        if mask is None:
            missing.append(image.name)
        else:
            pairs.append({"image": image.name, "mask": mask.name})
    if missing:
        raise ValueError(f"Missing masks in {dataset_dir.name}: {missing[:5]} ({len(missing)} total)")
    if not pairs:
        raise ValueError(f"No image/mask pairs in {dataset_dir}")
    return pairs


def make_split_manifest(data_root: Path, dataset_names: list[str], seed: int,
                        validation_fraction: float = 0.10) -> dict:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    datasets = {}
    for offset, name in enumerate(dataset_names):
        pairs = list_pairs(data_root / name)
        indices = np.arange(len(pairs))
        np.random.RandomState(seed + offset).shuffle(indices)
        val_count = max(1, int(np.ceil(len(pairs) * validation_fraction)))
        val_indices = set(indices[-val_count:].tolist())
        train = [pair for index, pair in enumerate(pairs) if index not in val_indices]
        validation = [pair for index, pair in enumerate(pairs) if index in val_indices]
        datasets[name] = {"train": train, "validation": validation,
                          "counts": {"all": len(pairs), "train": len(train), "validation": len(validation)}}
    manifest = {"seed": seed, "validation_fraction": validation_fraction, "datasets": datasets}
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["sha256"] = hashlib.sha256(canonical).hexdigest()
    return manifest


def save_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
