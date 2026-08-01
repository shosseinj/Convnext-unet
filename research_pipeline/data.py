"""Lazy paired-image datasets using the audited preprocessing contract."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class ManifestSegmentationDataset(Dataset):
    def __init__(self, data_root: Path, dataset_name: str, pairs: list[dict],
                 input_size: int = 352, training: bool = False, augmentation: dict | None = None):
        self.image_dir = data_root / dataset_name / "images"
        self.mask_dir = data_root / dataset_name / "masks"
        self.dataset_name = dataset_name
        self.pairs = pairs
        self.input_size = input_size
        self.training = training
        self.augmentation = augmentation or {}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        image = cv2.imread(str(self.image_dir / pair["image"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.mask_dir / pair["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Unreadable pair: {self.dataset_name}/{pair}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy((mask > 127).astype("float32")).unsqueeze(0)
        if self.training:
            image, mask = self._augment(image, mask)
        return image.clamp(0, 1), (mask > 0.5).float(), pair["image"]

    def _augment(self, image, mask):
        cfg = self.augmentation
        if random.random() < cfg.get("horizontal_flip_probability", 0):
            image, mask = torch.flip(image, [-1]), torch.flip(mask, [-1])
        if random.random() < cfg.get("vertical_flip_probability", 0):
            image, mask = torch.flip(image, [-2]), torch.flip(mask, [-2])
        if random.random() < cfg.get("rotation_probability", 0):
            angle = random.uniform(-cfg.get("rotation_degrees", 15), cfg.get("rotation_degrees", 15))
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        if random.random() < cfg.get("crop_probability", 0):
            low, high = cfg.get("crop_ratio", [0.7, 1.0])
            height, width = image.shape[-2:]
            ratio = random.uniform(low, high)
            crop_h, crop_w = max(1, int(height * ratio)), max(1, int(width * ratio))
            top, left = random.randint(0, height - crop_h), random.randint(0, width - crop_w)
            image = TF.resize(TF.crop(image, top, left, crop_h, crop_w), [height, width],
                              interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(TF.crop(mask, top, left, crop_h, crop_w), [height, width],
                             interpolation=TF.InterpolationMode.NEAREST)
        if random.random() < cfg.get("photometric_probability", 0):
            image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
            image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))
            image = TF.adjust_saturation(image, random.uniform(0.9, 1.1))
        if random.random() < cfg.get("blur_probability", 0):
            image = TF.gaussian_blur(image, 3)
        if random.random() < cfg.get("noise_probability", 0):
            image = image + torch.randn_like(image) * cfg.get("noise_std", 0.015)
        return image, mask
