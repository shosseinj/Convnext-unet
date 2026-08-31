import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse
import random
import sys
from pathlib import Path
import pickle as pkl
import numpy as np

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from utils_torch import *  # You'll need to adapt utils for PyTorch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)

from PIL import Image
from sklearn.model_selection import train_test_split
import glob
import cv2
from ablation_cli import add_ablation_arguments
from ablation_registry import get_experiment
from amp_training import (
    add_scaler_state, autocast_context, create_grad_scaler,
    finish_optimizer_step, restore_scaler_state,
)
from checkpoint_management import (
    atomic_save_best, atomic_save_checkpoint, load_checkpoint_file,
    mark_training_complete, prepare_best_checkpoint, prepare_checkpoint,
)
from training_artifacts import append_history_row, write_training_summary


import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os 

import os
import glob
import cv2
import numpy as np

from sklearn.model_selection import train_test_split


class Dataset:

    def __init__(
        self,
        data_name,
        logging_dir,
        flatten,
        ttfs_convert,
        ttfs_noise=0,
        data_path='./dataset/',
        input_size=(256, 256),
        evaluate_dataset='kvasir'
    ):

        self.name = data_name
        self.flatten = flatten
        self.data_path = data_path
        self.noise = ttfs_noise
        self.input_size = input_size
        self.logging_dir = logging_dir
        self.evaluate_dataset = evaluate_dataset


        # disable OpenCV warnings
        try:
            cv2.utils.logging.setLogLevel(
                cv2.utils.logging.LOG_LEVEL_ERROR
            )
        except:
            pass


        self.get_features_vectors()


        self.ttfss_convert = ttfs_convert

        if ttfs_convert:
            self.convert_ttfs()


    def get_features_vectors(self):

        """
        Train: 90% of Kvasir-SEG + 90% of CVC-ClinicDB (combined)
        Val: 10% of Kvasir-SEG (separate) + 10% of CVC-ClinicDB (separate)
        Test: CVC-300, ETIS-LARIBPOLYPDB, CVC-ColonDB (each separately)
        """

        self.num_of_classes = 2

        self.input_shape = (
            3,
            self.input_size[0],
            self.input_size[1]
        )

        extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
        
        # ==========================================
        # 1. Load Kvasir-SEG
        # ==========================================
        print("\n" + "="*60)
        print("Loading Kvasir-SEG...")
        print("="*60)
        
        kvasir_images = []
        kvasir_masks = []
        
        kvasir_image_dir = os.path.join(self.data_path, "Kvasir-SEG", "images")
        kvasir_mask_dir = os.path.join(self.data_path, "Kvasir-SEG", "masks")
        
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(kvasir_image_dir, ext)))
        image_files = sorted(image_files)
        
        for img_path in image_files:
            filename = os.path.basename(img_path)
            mask_path = os.path.join(kvasir_mask_dir, filename)
            
            if not os.path.exists(mask_path):
                continue
            
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None or mask is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)
            
            img = (img.astype(np.float32) / 255.0)
            img = np.transpose(img, (2, 0, 1))
            mask = (mask > 127).astype(np.float32)
            
            kvasir_images.append(img)
            kvasir_masks.append(mask)
        
        print(f"Loaded Kvasir-SEG: {len(kvasir_images)} images")
        
        # ==========================================
        # 2. Load CVC-ClinicDB
        # ==========================================
        print("\n" + "="*60)
        print("Loading CVC-ClinicDB...")
        print("="*60)
        
        clinicdb_images = []
        clinicdb_masks = []
        
        clinicdb_image_dir = os.path.join(self.data_path, "CVC-ClinicDB", "images")
        clinicdb_mask_dir = os.path.join(self.data_path, "CVC-ClinicDB", "masks")
        
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(clinicdb_image_dir, ext)))
        image_files = sorted(image_files)
        
        for img_path in image_files:
            filename = os.path.basename(img_path)
            mask_path = os.path.join(clinicdb_mask_dir, filename)
            
            if not os.path.exists(mask_path):
                continue
            
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None or mask is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)
            
            img = (img.astype(np.float32) / 255.0)
            img = np.transpose(img, (2, 0, 1))
            mask = (mask > 127).astype(np.float32)
            
            clinicdb_images.append(img)
            clinicdb_masks.append(mask)
        
        print(f"Loaded CVC-ClinicDB: {len(clinicdb_images)} images")
        
        # ==========================================
        # 3. Split datasets
        # ==========================================
        from sklearn.model_selection import train_test_split
        
        # Kvasir split (90% train, 10% val)
        kvasir_x_train, kvasir_x_val, kvasir_y_train, kvasir_y_val = train_test_split(
            np.array(kvasir_images, dtype=np.float32),
            np.array(kvasir_masks, dtype=np.float32),
            test_size=0.1,
            random_state=42,
            shuffle=True
        )
        
        # ClinicDB split (90% train, 10% val)
        clinicdb_x_train, clinicdb_x_val, clinicdb_y_train, clinicdb_y_val = train_test_split(
            np.array(clinicdb_images, dtype=np.float32),
            np.array(clinicdb_masks, dtype=np.float32),
            test_size=0.1,
            random_state=42,
            shuffle=True
        )
        
        # Combine training data
        self.x_train = np.concatenate([kvasir_x_train, clinicdb_x_train], axis=0)
        self.y_train = np.concatenate([kvasir_y_train, clinicdb_y_train], axis=0)
        
        # Shuffle combined training data
        indices = np.random.permutation(len(self.x_train))
        self.x_train = self.x_train[indices].astype(np.float32)
        self.y_train = self.y_train[indices].astype(np.float32)
        

        eveluate_dataset = self.evaluate_dataset
        if eveluate_dataset == 'both':
            self.x_test = np.concatenate(
                [
                    kvasir_x_val,
                    clinicdb_x_val
                ],
                axis=0
            )

            self.y_test = np.concatenate(
                [
                    kvasir_y_val,
                    clinicdb_y_val
                ],
                axis=0
            )



            # Shuffle test
            idx = np.random.permutation(len(self.x_test))

            self.x_test = self.x_test[idx].astype(np.float32)
            self.y_test = self.y_test[idx].astype(np.float32)


        elif eveluate_dataset== 'kvasir':
            self.x_test = kvasir_x_val.astype(np.float32)
            self.y_test = kvasir_y_val.astype(np.float32)

        elif eveluate_dataset== 'clinicdb':
            self.x_test = clinicdb_x_val.astype(np.float32)
            self.y_test = clinicdb_y_val.astype(np.float32)

        else:

            test_images = []
            test_masks = []
            
            test_image_dir = os.path.join(self.data_path, eveluate_dataset, "images")
            test_mask_dir = os.path.join(self.data_path, eveluate_dataset, "masks")

            image_files = []
            for ext in extensions:
                image_files.extend(glob.glob(os.path.join(test_image_dir, ext)))
            image_files = sorted(image_files)
            
            for img_path in image_files:
                filename = os.path.basename(img_path)
                mask_path = os.path.join(test_mask_dir, filename)
                
                if not os.path.exists(mask_path):
                    continue
                
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None or mask is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)
                
                img = (img.astype(np.float32) / 255.0)
                img = np.transpose(img, (2, 0, 1))
                mask = (mask > 127).astype(np.float32)
                
                test_images.append(img)
                test_masks.append(mask)
  
            self.x_test = np.array(test_images, dtype=np.float32)
            self.y_test =  np.array(test_masks, dtype=np.float32)


                
        
def grad_norm_except_encoder(model):
    params = [
        p for name, p in model.named_parameters()
        if "encoder" not in name and p.grad is not None
    ]

    total_norm = torch.norm(
        torch.stack([p.grad.norm(2) for p in params]),
        2
    )

    return total_norm.item()

def mixup_data(images, masks, alpha=0.2):
    """Mix two training samples together"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    B = images.size(0)
    idx = torch.randperm(B)

    mixed_images = lam * images + (1 - lam) * images[idx]
    mixed_masks  = lam * masks  + (1 - lam) * masks[idx]

    return mixed_images, mixed_masks

bce = nn.BCEWithLogitsLoss()


DEEP_SUPERVISION_WEIGHTS = (1.0, 0.1, 0.05, 0.02)


def _output_list(outputs):
    if isinstance(outputs, dict):
        return list(outputs.values())
    if isinstance(outputs, (list, tuple)):
        return list(outputs)
    return [outputs]


def _resize_to_target(logits, target):
    if logits.shape != target.shape:
        logits = F.interpolate(
            logits,
            size=target.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
    return logits


def _outputs_are_finite(outputs):
    return all(torch.isfinite(output).all() for output in _output_list(outputs))


def deep_supervision_loss(outputs, target, criterion, weights=DEEP_SUPERVISION_WEIGHTS):
    if isinstance(outputs, dict):
        from research_pipeline.losses import ugbr_composite_loss
        components = ugbr_composite_loss(outputs, target, criterion)
        return _resize_to_target(outputs["final_logits"], target), components["total"]
    outputs = _output_list(outputs)
    main_output = _resize_to_target(outputs[0], target)
    total_loss = criterion(main_output, target)

    for aux_output, weight in zip(outputs[1:], weights[1:]):
        aux_output = _resize_to_target(aux_output, target)
        total_loss = total_loss + weight * criterion(aux_output, target)

    return main_output, total_loss


def deep_supervision_weights_for_epoch(
    epoch_number, schedule="constant", anneal_start=96, anneal_end=128
):
    """Return main/d2/d3 weights for a one-based training epoch."""
    if schedule == "constant":
        return DEEP_SUPERVISION_WEIGHTS
    if anneal_start <= 0 or anneal_end <= anneal_start:
        raise ValueError("Deep-supervision annealing requires 0 < start < end")
    if epoch_number <= anneal_start:
        scale = 1.0
    elif epoch_number >= anneal_end:
        scale = 0.0
    else:
        scale = (anneal_end - epoch_number) / (anneal_end - anneal_start)
    return (1.0, 0.1 * scale, 0.05 * scale, 0.02 * scale)


def lesion_size_sampling_weights(masks):
    """Assign sampling weights based on foreground area ratios."""
    ratios = np.asarray(masks, dtype=np.float32).reshape(len(masks), -1).mean(axis=1)
    weights = np.ones(len(ratios), dtype=np.float64)
    weights[(ratios < 0.10) & (ratios >= 0.02)] = 2.0
    weights[ratios < 0.02] = 4.0
    return torch.as_tensor(weights, dtype=torch.double), ratios


def frequency_augmentation_probability(
    epoch_number, total_epochs, maximum=0.5,
    constant_fraction=0.70, anneal_end_fraction=0.80,
):
    if not 0 <= maximum <= 1:
        raise ValueError("Frequency augmentation probability must be in [0, 1]")
    if not 0 < constant_fraction < anneal_end_fraction <= 1:
        raise ValueError("Frequency schedule requires 0 < constant < anneal_end <= 1")
    constant_end = int(round(total_epochs * constant_fraction))
    anneal_end = int(round(total_epochs * anneal_end_fraction))
    if epoch_number <= constant_end:
        return maximum
    if epoch_number >= anneal_end:
        return 0.0
    return maximum * (anneal_end - epoch_number) / (anneal_end - constant_end)


def frequency_style_augment(
    images, probability, generator, max_mix=0.5,
    region_min=0.01, region_max=0.05,
):
    """Mix only low-frequency amplitudes while preserving source phase."""
    if not 0 <= max_mix <= 1:
        raise ValueError("Frequency amplitude mixing must be in [0, 1]")
    if not 0 < region_min <= region_max <= 0.5:
        raise ValueError("Frequency region requires 0 < min <= max <= 0.5")
    if probability <= 0 or images.shape[0] < 2:
        return images
    if torch.rand((), generator=generator).item() >= probability:
        return images
    batch, _, height, width = images.shape
    permutation = torch.randperm(batch, generator=generator).to(images.device)
    region_ratio = region_min + (region_max - region_min) * torch.rand(
        (), generator=generator
    ).item()
    mix = max_mix * torch.rand((), generator=generator).item()
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(images.float(), dim=(-2, -1)), dim=(-2, -1)
    )
    amplitude = spectrum.abs()
    phase = torch.angle(spectrum)
    mixed_amplitude = amplitude.clone()
    radius_h = max(1, int(height * region_ratio))
    radius_w = max(1, int(width * region_ratio))
    center_h, center_w = height // 2, width // 2
    region = (
        slice(None), slice(None),
        slice(center_h - radius_h, center_h + radius_h + 1),
        slice(center_w - radius_w, center_w + radius_w + 1),
    )
    mixed_amplitude[region] = (
        (1.0 - mix) * amplitude[region] + mix * amplitude[permutation][region]
    )
    mixed_spectrum = torch.polar(mixed_amplitude, phase)
    augmented = torch.fft.ifft2(
        torch.fft.ifftshift(mixed_spectrum, dim=(-2, -1)), dim=(-2, -1)
    ).real
    return augmented.to(images.dtype).clamp_(0.0, 1.0)


def soft_dice_coefficient(logits, target, smooth=1.0):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    target = target.float().view(target.size(0), -1)
    intersection = (probs * target).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        probs.sum(dim=1) + target.sum(dim=1) + smooth
    )
    return dice.mean().item()


def encoder_stage_name(stages_unfrozen):
    if stages_unfrozen <= 0:
        return "decoder"
    if stages_unfrozen >= 5:
        return "all"
    return f"encoder_last_{stages_unfrozen}"


def advance_encoder_unfreezing(
    stages_unfrozen, plateau_epochs, improved, patience=10
):
    if improved:
        return stages_unfrozen, 0, False
    if stages_unfrozen >= 5:
        return stages_unfrozen, plateau_epochs, False
    plateau_epochs += 1
    if plateau_epochs < patience:
        return stages_unfrozen, plateau_epochs, False
    return stages_unfrozen + 1, 0, True


FIXED_UNFREEZE_EPOCHS = (16, 46, 76, 106, 136)


def fixed_encoder_stages_for_epoch(epoch_number, milestones=FIXED_UNFREEZE_EPOCHS):
    """Return cumulative encoder exposure for a one-based epoch number."""
    return sum(epoch_number >= milestone for milestone in milestones)


def parse_fixed_unfreeze_epochs(value):
    milestones = tuple(int(item.strip()) for item in value.split(',') if item.strip())
    if len(milestones) != 5 or any(epoch <= 0 for epoch in milestones):
        raise ValueError("--fixed_unfreeze_epochs requires five positive epochs")
    if tuple(sorted(set(milestones))) != milestones:
        raise ValueError("--fixed_unfreeze_epochs must be strictly increasing")
    return milestones


def create_plateau_scheduler(optimizer, min_lr, patience=5, factor=0.9):
    return ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=factor,
        patience=patience,
        threshold=1e-4,
        min_lr=min_lr,
    )


def grad_clip_norm_for_epoch(epoch, encoder_frozen_epochs):
    return 5.0 if epoch < encoder_frozen_epochs else 3.5


def set_training_stage(model, stage):
    decoder_prefixes = (
        "detail.",
        "detail_branch.",
        "detail_conv.",
        "detail_fusion.",
        "final_refine.",
        "decoder",
        "bsei",
        "aux",
        "bottleneck.",
        "context.",
        "ugbr.",
        "csaf.",
        "fafem.",
        "fafem_stage1.",
        "fafem_stage2.",
        "fafem_stage3.",
        "cross_level_fusion.",
        "geometry_conv_stage3.",
        "mscb_lite_stage3.",
        "mscb_lite_stage2.",
        "mscb_lite_stage1.",
        "fg_mscb_lite_stage3.",
        "residual_fg_mscb_lite_stage3.",
        "lka_lite_stage3.",
        "uncertainty_refinement.",
    )
    if stage == "detail":
        trainable_prefixes = ("detail.", "detail_branch.", "detail_conv.",
                              "detail_fusion.", "final_refine.")
    elif stage == "decoder":
        trainable_prefixes = decoder_prefixes
    elif stage.startswith("encoder_last_"):
        stages_to_unfreeze = int(stage.rsplit("_", 1)[1])
        first_stage = 4 - stages_to_unfreeze
        encoder_prefixes = tuple(
            prefix
            for stage_index in range(first_stage, 4)
            for prefix in (
                (f"encoder.stages.{stage_index}.",)
                if stage_index == 0
                else (
                    f"encoder.downsample_layers.{stage_index}.",
                    f"encoder.stages.{stage_index}.",
                )
            )
        )
        trainable_prefixes = decoder_prefixes + encoder_prefixes
    elif stage == "all":
        trainable_prefixes = None
    else:
        raise ValueError(f"Unknown training stage: {stage}")

    trainable_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        param.requires_grad = trainable_prefixes is None or name.startswith(trainable_prefixes)
        if param.requires_grad:
            trainable_params += param.numel()

    return trainable_params, total_params


def set_frozen_modules_eval(module):
    for child in module.children():
        set_frozen_modules_eval(child)
        params = list(child.parameters(recurse=True))
        if params and not any(param.requires_grad for param in params):
            child.eval()
    if (
        hasattr(module, "downsample_layers")
        and hasattr(module, "stages")
        and hasattr(module, "dropout")
    ):
        encoder_params = list(module.parameters())
        if encoder_params and not all(
            parameter.requires_grad for parameter in encoder_params
        ):
            module.dropout.eval()


def optimizer_lr(optimizer, group_name, fallback_idx=0):
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return group["lr"]
    return optimizer.param_groups[fallback_idx]["lr"]


def encoder_optimizer_lr(optimizer):
    """Return the highest encoder LR for either optimizer profile."""
    for group_name in ("encoder_stage4", "encoder"):
        for group in optimizer.param_groups:
            if group.get("name") == group_name:
                return group["lr"]
    return optimizer.param_groups[0]["lr"]


def build_optimizer_param_groups(
    model, decoder_lr, encoder_lr, refine_lr, decoder_weight_decay,
    encoder_weight_decay, refine_weight_decay, profile="standard",
    encoder_layer_decay=0.8,
):
    """Build exhaustive, non-overlapping AdamW groups for the selected recipe."""
    if profile not in {"standard", "layerwise_convnext"}:
        raise ValueError(f"Unsupported optimizer profile: {profile}")
    if not 0 < encoder_layer_decay <= 1:
        raise ValueError("encoder_layer_decay must be in (0, 1]")

    grouped = {}
    group_settings = {
        "refine": (refine_lr, refine_weight_decay),
        "decoder": (decoder_lr, decoder_weight_decay),
    }
    if profile == "standard":
        group_settings["encoder"] = (encoder_lr, encoder_weight_decay)
    else:
        group_settings.update({
            "encoder_stage4": (encoder_lr, encoder_weight_decay),
            "encoder_stage3": (encoder_lr * encoder_layer_decay, encoder_weight_decay),
            "encoder_stage2": (encoder_lr * encoder_layer_decay ** 2, encoder_weight_decay),
            "encoder_stage1_stem": (
                encoder_lr * encoder_layer_decay ** 3, encoder_weight_decay
            ),
        })

    for name, parameter in model.named_parameters():
        if name.startswith("final_refine."):
            group_name = "refine"
        elif not name.startswith("encoder."):
            group_name = "decoder"
        elif profile == "standard":
            group_name = "encoder"
        else:
            group_name = None
            for stage_index, candidate in (
                (3, "encoder_stage4"), (2, "encoder_stage3"),
                (1, "encoder_stage2"), (0, "encoder_stage1_stem"),
            ):
                if name.startswith((
                    f"encoder.stages.{stage_index}.",
                    f"encoder.downsample_layers.{stage_index}.",
                )):
                    group_name = candidate
                    break
            if group_name is None:
                raise ValueError(f"Unmapped ConvNeXt encoder parameter: {name}")
        grouped.setdefault(group_name, []).append(parameter)

    missing = [name for name in group_settings if not grouped.get(name)]
    if missing:
        raise ValueError(f"Optimizer groups contain no parameters: {missing}")
    groups = []
    for name, (learning_rate, weight_decay) in group_settings.items():
        groups.append({
            "name": name,
            "params": grouped[name],
            "lr": learning_rate,
            "weight_decay": weight_decay,
        })
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("A parameter was assigned to more than one optimizer group")
    if len(parameter_ids) != sum(1 for _ in model.parameters()):
        raise ValueError("Optimizer grouping did not cover every model parameter")
    return groups


def create_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr):
    if warmup_epochs <= 0 or warmup_epochs >= total_epochs:
        raise ValueError("warmup_cosine requires 0 < warmup_epochs < epochs")
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0,
                total_iters=warmup_epochs,
            ),
            CosineAnnealingLR(
                optimizer, T_max=max(total_epochs - warmup_epochs, 1),
                eta_min=min_lr,
            ),
        ],
        milestones=[warmup_epochs],
    )


def summarize_fg_mscb_alpha(alpha):
    """Compute per-branch sample statistics over all FG-MSCB training samples."""
    if alpha.ndim != 2 or alpha.shape[1] != 3 or alpha.shape[0] < 2:
        raise ValueError("FG-MSCB alpha statistics require at least two [N, 3] samples")
    alpha = alpha.detach().to(dtype=torch.float64)
    mean = alpha.mean(dim=0)
    std = alpha.std(dim=0, correction=1)
    return {
        "alpha1_mean": mean[0].item(), "alpha1_std": std[0].item(),
        "alpha3_mean": mean[1].item(), "alpha3_std": std[1].item(),
        "alpha5_mean": mean[2].item(), "alpha5_std": std[2].item(),
        "sample_count": alpha.shape[0],
    }


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    encoder = sum(param.numel() for param in model.encoder.parameters()) if hasattr(model, "encoder") else 0
    return total, trainable, encoder


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.update(model, decay=0.0)

    def update(self, model, decay=None):
        decay = self.decay if decay is None else decay
        with torch.no_grad():
            for key, value in model.state_dict().items():
                if not torch.is_floating_point(value):
                    continue
                value = value.detach()
                if key not in self.shadow:
                    self.shadow[key] = value.clone()
                else:
                    self.shadow[key].mul_(decay).add_(value, alpha=1.0 - decay)

    def store(self, model):
        self.backup = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if key in self.shadow
        }

    def copy_to(self, model):
        state = model.state_dict()
        with torch.no_grad():
            for key, value in self.shadow.items():
                if key in state:
                    state[key].copy_(value.to(device=state[key].device, dtype=state[key].dtype))

    def restore(self, model):
        state = model.state_dict()
        with torch.no_grad():
            for key, value in self.backup.items():
                if key in state:
                    state[key].copy_(value.to(device=state[key].device, dtype=state[key].dtype))
        self.backup = {}

    def state_dict(self):
        return {key: value.detach().cpu().clone() for key, value in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {
            key: value.detach().clone()
            for key, value in state_dict.items()
            if torch.is_floating_point(value)
        }


def snapshot_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def snapshot_rng_state():
    state = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(checkpoint):
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
        )


def scored_model_state_dict(model, ema=None):
    if ema is None:
        return snapshot_state_dict(model)

    ema.store(model)
    ema.copy_to(model)
    state = snapshot_state_dict(model)
    ema.restore(model)
    return state


def train_epoch_segmentation(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    scheduler,
    threshold,
    ema=None,
    amp_enabled=False,
    scaler=None,
    max_grad_norm=3.5,
    deep_supervision_weights=DEEP_SUPERVISION_WEIGHTS,
    frequency_augmentation_probability_value=0.0,
    frequency_generator=None,
    frequency_max_mix=0.5,
    frequency_region_min=0.01,
    frequency_region_max=0.05,
    collect_fg_mscb_alpha=False,
):
    import random
    import numpy as np
    import logging
    from tqdm import tqdm
    import torch
    import torch.nn.functional as F

    model.train()
    set_frozen_modules_eval(model)

    running_loss = 0.0
    running_dice = 0.0
    running_soft_dice = 0.0

    skipped_batches = 0

    # ----------------------------
    # Diagnostics
    # ----------------------------
    enc_grad_sum = 0.0
    dec_grad_sum = 0.0
    geometry_grad_sum = 0.0
    enc_grad_rms_sum = 0.0
    dec_grad_rms_sum = 0.0
    geometry_grad_rms_sum = 0.0
    prob_mean_sum = 0.0
    prob_std_sum = 0.0
    mask_area_sum = 0.0

    valid_batches = 0
    fg_alpha_samples = {}

    def frequency_guided_modules():
        modules = {
            "stage1": getattr(model, "residual_fg_mscb_lite_stage1", None),
            "stage2": getattr(model, "residual_fg_mscb_lite_stage2", None),
            "stage3": (
                getattr(model, "residual_fg_mscb_lite_stage3_skip", None) or
                getattr(model, "fg_mscb_lite_stage3", None) or
                getattr(model, "residual_fg_mscb_lite_stage3", None) or
                getattr(model, "deformable_residual_fg_mscb_lite_stage3", None) or
                getattr(model, "partial_deformable_residual_fg_mscb_lite_stage3", None) or
                getattr(model, "f4_f3_context_guided_mscb_lite_stage3", None)
            ),
        }
        return {name: module for name, module in modules.items() if module is not None}

    def grad_norm(module):
        total = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total += p.grad.detach().norm(2).item() ** 2
        return total ** 0.5

    pbar = tqdm(train_loader, desc="Training", file=sys.stdout, dynamic_ncols=True)

    for batch_idx, (data, target) in enumerate(pbar):

        data = data.to(device)
        target = target.to(device)

        data = torch.clamp(data, 0.0, 1.0)
        if frequency_augmentation_probability_value > 0:
            data = frequency_style_augment(
                data,
                frequency_augmentation_probability_value,
                frequency_generator,
                max_mix=frequency_max_mix,
                region_min=frequency_region_min,
                region_max=frequency_region_max,
            )

        if target.dim() == 3:
            target = target.unsqueeze(1)

        # MixUp
        if random.random() < 0.2 and False:
            lam = np.random.beta(0.2, 0.2)
            idx = torch.randperm(data.size(0), device=device)

            data = lam * data + (1 - lam) * data[idx]
            target = lam * target + (1 - lam) * target[idx]

        optimizer.zero_grad()

        # ----------------------------
        # Forward
        # ----------------------------
        with autocast_context(amp_enabled, device.type):
            outputs = model(data)

            if not _outputs_are_finite(outputs):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            output, loss = deep_supervision_loss(
                outputs, target, criterion, weights=deep_supervision_weights
            )

        if not torch.isfinite(loss):
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        if collect_fg_mscb_alpha:
            for stage_name, fg_mscb in frequency_guided_modules().items():
                alpha = fg_mscb.last_alpha
                if alpha is None:
                    raise RuntimeError(f"FG-MSCB alpha was not produced for {stage_name}")
                fg_alpha_samples.setdefault(stage_name, []).append(
                    alpha.detach().to(device="cpu", dtype=torch.float64)
                )

        # ----------------------------
        # Backward
        # ----------------------------
        try:
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
        except RuntimeError:
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        # NaN gradient detection
        bad_grad = False
        for p in model.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break

        if bad_grad:
            skipped_batches += 1
            finish_optimizer_step(
                optimizer, scaler, amp_enabled, gradients_are_finite=False
            )
            continue

        # ----------------------------
        # Diagnostics BEFORE clipping
        # ----------------------------

        enc_grad = grad_norm(model.encoder)
        dec_grad = grad_norm_except_encoder(model)
        geometry_module = getattr(model, "geometry_conv_stage3", None)
        geometry_grad = grad_norm(geometry_module) if geometry_module is not None else 0.0
        from one_seed_models import gradient_rms
        enc_grad_rms = gradient_rms(model.encoder.parameters())
        encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
        dec_grad_rms = gradient_rms(
            parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
        )
        geometry_grad_rms = (
            gradient_rms(geometry_module.parameters())
            if geometry_module is not None else 0.0
        )

        grad_norm_total = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_grad_norm
        )

        finish_optimizer_step(
            optimizer, scaler, amp_enabled, gradients_are_finite=True
        )
        if ema is not None:
            ema.update(model)

        # scheduler.step()

        with torch.no_grad():

            prob = torch.sigmoid(output)

            pred = (prob > threshold).float()

            dice = dice_coefficient(pred, target)
            soft_dice = soft_dice_coefficient(output, target)

            prob_mean = prob.mean().item()
            prob_std = prob.std().item()
            mask_area = pred.mean().item()

        # ----------------------------
        # Accumulate
        # ----------------------------

        running_loss += loss.item()
        running_dice += dice
        running_soft_dice += soft_dice

        enc_grad_sum += enc_grad
        dec_grad_sum += dec_grad
        geometry_grad_sum += geometry_grad
        enc_grad_rms_sum += enc_grad_rms
        dec_grad_rms_sum += dec_grad_rms
        geometry_grad_rms_sum += geometry_grad_rms

        prob_mean_sum += prob_mean
        prob_std_sum += prob_std
        mask_area_sum += mask_area

        valid_batches += 1

        # Current learning rates

        refine_lr = optimizer_lr(optimizer, "refine", fallback_idx=0)
        enc_lr = encoder_optimizer_lr(optimizer)
        dec_lr = optimizer_lr(optimizer, "decoder", fallback_idx=-1)

        pbar.set_postfix(

            Loss=f"{loss.item():.4f}",

            Dice=f"{dice:.4f}",

            SoftDice=f"{soft_dice:.4f}",

            EncGrad=f"{enc_grad:.2f}",

            DecGrad=f"{dec_grad:.2f}",

            GeoGrad=f"{geometry_grad:.2f}",

            GradClip=f"{max_grad_norm:.1f}",

            EncRMS=f"{enc_grad_rms:.2e}",

            DecRMS=f"{dec_grad_rms:.2e}",

            GeoRMS=f"{geometry_grad_rms:.2e}",

            Prob=f"{prob_mean:.3f}",

            Area=f"{mask_area:.3f}",

            RefineLR=f"{refine_lr:.2e}",

            EncLR=f"{enc_lr:.2e}",

            DecLR=f"{dec_lr:.2e}",

            Skip=skipped_batches,
        )

    if skipped_batches > 0:
        logging.warning(
            f"Skipped {skipped_batches}/{len(train_loader)} batches."
        )



    train_loss = running_loss / max(valid_batches, 1)
    if not collect_fg_mscb_alpha:
        return train_loss
    if not fg_alpha_samples:
        raise RuntimeError("FG-MSCB alpha was not collected during training")
    statistics_by_stage = {}
    modules = frequency_guided_modules()
    for stage_name, samples in fg_alpha_samples.items():
        statistics = summarize_fg_mscb_alpha(torch.cat(samples, dim=0))
        statistics["guidance_strength"] = (
            modules[stage_name].effective_guidance_strength().detach().item()
        )
        statistics_by_stage[stage_name] = statistics
        logging.info(
            "%s FG-MSCB alpha over %d training samples: "
            "a1=%.6f +/- %.6f | a3=%.6f +/- %.6f | a5=%.6f +/- %.6f | strength=%.6f",
            stage_name.capitalize(), statistics["sample_count"],
            statistics["alpha1_mean"], statistics["alpha1_std"],
            statistics["alpha3_mean"], statistics["alpha3_std"],
            statistics["alpha5_mean"], statistics["alpha5_std"],
            statistics["guidance_strength"],
        )
    if len(statistics_by_stage) == 1 and "stage3" in statistics_by_stage:
        return train_loss, statistics_by_stage["stage3"]
    return train_loss, {
        f"{stage_name}_{name}": value
        for stage_name, statistics in statistics_by_stage.items()
        for name, value in statistics.items()
    }

def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient."""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()  # Return as Python float


def iou_score(pred, target, smooth=1e-6):
    """Calculate IoU (Jaccard) score."""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()  # Return as Python float


# def evaluate_with_tta(model, test_loader, device, threshold=0.40):
#     model.eval()
#     all_preds = []
#     all_targets = []
    
#     with torch.no_grad():
#         for data, target in tqdm(test_loader, desc='TTA'):
#             data = data.to(device)
            
#             # Original
#             pred1 = torch.sigmoid(model(data))
            
#             # Horizontal flip
#             pred2 = torch.flip(torch.sigmoid(model(torch.flip(data, [-1]))), [-1])
            
#             # Vertical flip  
#             pred3 = torch.flip(torch.sigmoid(model(torch.flip(data, [-2]))), [-2])
            
#             # Average
#             pred = (pred1 + pred2 + pred3) / 3
            
#             all_preds.append((pred > threshold).float().cpu())
#             all_targets.append(target.cpu())
    
#     # Calculate metrics
#     all_preds = torch.cat(all_preds)
#     all_targets = torch.cat(all_targets)
    
#     dice = dice_coefficient(all_preds, all_targets)
#     iou = iou_score(all_preds, all_targets)
    
#     return dice, iou

def evaluate_with_tta(model, test_loader, device, threshold=0.40):
    model.eval()
    all_preds = []
    all_targets = []
    
    # Initialize advanced metric calculators
    SM = py_sod_metrics.Smeasure()
    WFM = py_sod_metrics.WeightedFmeasure()
    EM = py_sod_metrics.Emeasure()
    MAE = py_sod_metrics.MAE()
    
    with torch.no_grad():
        for data, target in tqdm(
            test_loader, desc='TTA', file=sys.stdout, dynamic_ncols=True
        ):
            data = data.to(device)
            
            # Original
            pred1 = torch.sigmoid(model(data))
            
            # Horizontal flip
            pred2 = torch.flip(torch.sigmoid(model(torch.flip(data, [-1]))), [-1])
            
            # Vertical flip  
            pred3 = torch.flip(torch.sigmoid(model(torch.flip(data, [-2]))), [-2])
            
            # Average (Continuous Probability Map)
            prob = (pred1 + pred2 + pred3) / 3
            
            # Store thresholded predictions for Dice/IoU
            all_preds.append((prob > threshold).float().cpu())
            all_targets.append(target.cpu())
            
            # --- Advanced Metrics Evaluation FIX ---
            prob_np = prob.cpu().numpy()
            target_np = target.cpu().numpy()
            
            # Ensure target has a channel dimension (B, 1, H, W)
            if target_np.ndim == 3:
                target_np = np.expand_dims(target_np, axis=1)
                
            for b in range(prob_np.shape[0]):
                # 1. Scale continuous probability to [0, 255] and convert to uint8
                p = (prob_np[b, 0] * 255.0).astype(np.uint8) 
                g = (target_np[b, 0] * 255.0).astype(np.uint8)
                
                # 2. Ensure ground truth is strictly binary (0 or 255)
                g = (g > 127).astype(np.uint8) * 255
                
                # Step the calculators safely with uint8 data
                SM.step(pred=p, gt=g)
                WFM.step(pred=p, gt=g)
                EM.step(pred=p, gt=g)
                MAE.step(pred=p, gt=g)
    
    # Calculate basic metrics (Dice & IoU)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    dice = dice_coefficient(all_preds, all_targets)
    iou = iou_score(all_preds, all_targets)
    
    # Extract advanced metrics
    sm = SM.get_results()['sm']
    wfm = WFM.get_results()['wfm']
    em_max = EM.get_results()['em']['curve'].max()
    em_mean = EM.get_results()['em']['curve'].mean()
    mae = MAE.get_results()['mae']
    
    print(f"\nTTA Evaluation Results:")
    print(f"Dice: {dice:.4f} | IoU: {iou:.4f}")
    print(f"S-measure: {sm:.4f} | Weighted F-measure: {wfm:.4f}")
    print(f"Max E-measure: {em_max:.4f} | Mean E-measure: {em_mean:.4f} | MAE: {mae:.4f}\n")
    
    return dice, iou


# def test_segmentation(model, test_loader, criterion, device, threshold=0.3, use_tta=False):
#     """Testing function with optional TTA"""
#     model.eval()
#     test_loss = 0
#     dice_scores = []
#     iou_scores = []
    
#     with torch.no_grad():
#         for data, target in tqdm(test_loader, desc='Testing'):
#             data, target = data.to(device), target.to(device)
            
#             if target.dim() == 3:
#                 target = target.unsqueeze(1)
            
#             if use_tta:
#                 # Use TTA for prediction
#                 prob = tta_predict(
#                     model,
#                     data
#                 )

#                 pred = (
#                     prob > threshold
#                 ).float()
                
#                 # For loss calculation, use original forward pass
#                 output = model(data)
#                 if output.shape != target.shape:
#                     output = F.interpolate(output, size=target.shape[2:], mode='bilinear', align_corners=False)
#                 test_loss += criterion(output, target).item()
#             else:
#                 # Regular prediction
#                 output = model(data)
#                 if output.shape != target.shape:
#                     output = F.interpolate(output, size=target.shape[2:], mode='bilinear', align_corners=False)
#                 test_loss += criterion(output, target).item()
#                 pred = (torch.sigmoid(output) > threshold).float()
            
#             dice_scores.append(dice_coefficient(pred, target))
#             iou_scores.append(iou_score(pred, target))
    
#     avg_loss = test_loss / len(test_loader)
#     avg_dice = np.mean(dice_scores)
#     avg_iou = np.mean(iou_scores)
    
#     return avg_loss, avg_dice, avg_iou


import torch.nn.functional as F
from tqdm import tqdm
import py_sod_metrics  # Make sure this is imported

def test_segmentation(model, test_loader, criterion, device, threshold=0.3,
                      use_tta=False, amp_enabled=False):
    """Testing function with optional TTA and advanced SOD metrics"""
    model.eval()
    test_loss = 0
    dice_scores = []
    iou_scores = []
    
    # Initialize advanced metric calculators
    SM = py_sod_metrics.Smeasure()
    WFM = py_sod_metrics.WeightedFmeasure()
    EM = py_sod_metrics.Emeasure()
    MAE = py_sod_metrics.MAE()
    
    with torch.no_grad():
        for data, target in tqdm(
            test_loader, desc='Testing', file=sys.stdout, dynamic_ncols=True
        ):
            data, target = data.to(device), target.to(device)
            
            if target.dim() == 3:
                target = target.unsqueeze(1)
            
            if use_tta:
                # Use TTA for prediction
                prob = tta_predict(
                    model,
                    data
                )

                pred = (
                    prob > threshold
                ).float()
                
                # For loss calculation, use original forward pass
                with autocast_context(amp_enabled, device.type):
                    output = model(data)
                if isinstance(output, dict):
                    output = output["final_logits"]
                elif isinstance(output, (tuple, list)):
                    output = output[0]
                if output.shape != target.shape:
                    output = F.interpolate(output, size=target.shape[2:], mode='bilinear', align_corners=False)
                test_loss += criterion(output, target).item()
            else:
                # Regular prediction
                with autocast_context(amp_enabled, device.type):
                    output = model(data)
                if isinstance(output, dict):
                    output = output["final_logits"]
                elif isinstance(output, (tuple, list)):
                    output = output[0]
                if output.shape != target.shape:
                    output = F.interpolate(output, size=target.shape[2:], mode='bilinear', align_corners=False)
                test_loss += criterion(output, target).item()
                
                prob = torch.sigmoid(output)
                pred = (prob > threshold).float()
            
            dice_scores.append(dice_coefficient(pred, target))
            iou_scores.append(iou_score(pred, target))
            
            # --- Advanced Metrics Evaluation FIX ---
            prob_np = prob.cpu().numpy()
            target_np = target.cpu().numpy()
            
            for b in range(prob_np.shape[0]):
                # 1. Scale to [0, 255] and convert to uint8
                p = (prob_np[b, 0] * 255.0).astype(np.uint8) 
                g = (target_np[b, 0] * 255.0).astype(np.uint8)
                
                # 2. Ensure ground truth is strictly binary (0 or 255)
                g = (g > 127).astype(np.uint8) * 255
                
                # Step the calculators safely
                SM.step(pred=p, gt=g)
                WFM.step(pred=p, gt=g)
                EM.step(pred=p, gt=g)
                MAE.step(pred=p, gt=g)
    
    avg_loss = test_loss / len(test_loader)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    
    # Get final SOD metric scores
    sm = SM.get_results()['sm']
    wfm = WFM.get_results()['wfm']
    em_max = EM.get_results()['em']['curve'].max()
    em_mean = EM.get_results()['em']['curve'].mean()
    mae = MAE.get_results()['mae']
    
    # Optional: Print the results cleanly
    print(f"\nTest Results:")
    print(f"Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}")
    print(f"S-measure: {sm:.4f} | Weighted F-measure: {wfm:.4f}")
    print(f"Max E-measure: {em_max:.4f} | Mean E-measure: {em_mean:.4f} | MAE: {mae:.4f}")
    
    return avg_loss, avg_dice, avg_iou

def find_best_threshold(model, test_loader, criterion, device):
    model.eval()

    # thresholds = np.arange(0.89, 0.990, 0.005)
    thresholds = np.arange(0.4, 0.990, 0.05)
    # hossein
    # thresholds = np.arange(0.7, 0.999, 0.005)

    best_threshold = 0.5
    best_iou = 0.0
    best_dice = 0.0

    with torch.no_grad():

        for threshold in thresholds:

            dice_scores = []
            iou_scores = []
            test_loss = 0.0

            for data, target in test_loader:

                data = data.to(device)
                target = target.to(device)

                if target.dim() == 3:
                    target = target.unsqueeze(1)

                output = model(data)

                if output.shape != target.shape:
                    output = F.interpolate(
                        output,
                        size=target.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )

                test_loss += criterion(output, target).item()

                pred = (torch.sigmoid(output) > threshold).float()

                dice_scores.append(
                    dice_coefficient(pred, target)
                )

                iou_scores.append(
                    iou_score(pred, target)
                )

            avg_dice = np.mean(dice_scores)
            avg_iou = np.mean(iou_scores)

            print(
                f"Threshold={threshold:.2f} "
                f"Dice={avg_dice:.4f} "
                f"IoU={avg_iou:.4f}"
            )

            if avg_dice > best_dice:
                best_iou = avg_iou
                best_dice = avg_dice
                best_threshold = threshold

    print("\n========================")
    print(f"Best Threshold : {best_threshold:.2f}")
    print(f"Best Dice      : {best_dice:.4f}")
    print(f"Best IoU       : {best_iou:.4f}")
    print("========================\n")

    return best_threshold, best_dice, best_iou


def evaluate_model(
    model,
    loader,
    criterion,
    device,
    use_tta=False
):
    """
    Finds best threshold using Dice and evaluates.
    """

    model.eval()

    thresholds = np.arange(0.05, 0.96, 0.01)

    best_threshold = 0.5
    best_dice = 0.0

    # --------------------------------------------------
    # Find threshold maximizing Dice
    # --------------------------------------------------
    with torch.no_grad():

        for threshold in thresholds:

            dice_scores = []

            for data, target in loader:

                data = data.to(device)
                target = target.to(device)

                if target.dim() == 3:
                    target = target.unsqueeze(1)

                if use_tta:

                    pred1 = torch.sigmoid(model(data))

                    pred2 = torch.flip(
                        torch.sigmoid(
                            model(torch.flip(data, [-1]))
                        ),
                        [-1]
                    )

                    pred3 = torch.flip(
                        torch.sigmoid(
                            model(torch.flip(data, [-2]))
                        ),
                        [-2]
                    )

                    prob = (pred1 + pred2 + pred3) / 3

                else:

                    prob = torch.sigmoid(
                        model(data)
                    )

                pred = (prob > threshold).float()

                dice_scores.append(
                    dice_coefficient(
                        pred,
                        target
                    )
                )

            avg_dice = np.mean(dice_scores)

            if avg_dice > best_dice:
                best_dice = avg_dice
                best_threshold = threshold

    # --------------------------------------------------
    # Final evaluation using best threshold
    # --------------------------------------------------
    total_loss = 0.0
    dice_scores = []
    iou_scores = []

    with torch.no_grad():

        for data, target in tqdm(
            loader, desc="Evaluating", file=sys.stdout, dynamic_ncols=True
        ):

            data = data.to(device)
            target = target.to(device)

            if target.dim() == 3:
                target = target.unsqueeze(1)

            output = model(data)

            if output.shape != target.shape:
                output = F.interpolate(
                    output,
                    size=target.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            total_loss += criterion(
                output,
                target
            ).item()

            if use_tta:

                pred1 = torch.sigmoid(model(data))

                pred2 = torch.flip(
                    torch.sigmoid(
                        model(torch.flip(data, [-1]))
                    ),
                    [-1]
                )

                pred3 = torch.flip(
                    torch.sigmoid(
                        model(torch.flip(data, [-2]))
                    ),
                    [-2]
                )

                prob = (pred1 + pred2 + pred3) / 3

            else:

                prob = torch.sigmoid(output)

            pred = (
                prob > best_threshold
            ).float()

            dice_scores.append(
                dice_coefficient(
                    pred,
                    target
                )
            )

            iou_scores.append(
                iou_score(
                    pred,
                    target
                )
            )

    return {
        "threshold": best_threshold,
        "loss": total_loss / len(loader),
        "dice": np.mean(dice_scores),
        "iou": np.mean(iou_scores)
    }

# Training function
from tqdm import tqdm





def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a = y
    y_b = y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



import math


class DiceBCELoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5, smooth=1e-6):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.smooth = smooth
        
    def forward(self, pred, target):
        # BCE with label smoothing
        target_smooth = target * 0.9 + 0.05  # Label smoothing
        bce = F.binary_cross_entropy_with_logits(pred, target_smooth)
        
        # Dice loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice = 1 - (2. * intersection + self.smooth) / (pred_sigmoid.sum() + target.sum() + self.smooth)
        
        return self.weight_bce * bce + self.weight_dice * dice


import albumentations as A
from albumentations.pytorch import ToTensorV2

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import random


import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class KvasirSEGDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, is_train=True, target_size=352):
        self.images = images
        self.masks = masks
        self.is_train = is_train
        self.target_size = target_size

    def __len__(self):
        return len(self.images)

    def _gaussian_blur_manual(self, tensor, kernel_size=9, sigma=4):
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()

        kernel_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0)

        padding = kernel_size // 2

        return F.conv2d(
            tensor,
            kernel.to(tensor.device),
            padding=padding
        )

    def _elastic_transform(self, image, mask,
                           alpha=150,
                           sigma=10):

        shape = image.shape[1:]

        dx = torch.randn(*shape) * sigma
        dy = torch.randn(*shape) * sigma

        dx = self._gaussian_blur_manual(
            dx.unsqueeze(0).unsqueeze(0),
            kernel_size=9,
            sigma=4
        )[0, 0]

        dy = self._gaussian_blur_manual(
            dy.unsqueeze(0).unsqueeze(0),
            kernel_size=9,
            sigma=4
        )[0, 0]

        grid_y, grid_x = torch.meshgrid(
            torch.arange(shape[0]),
            torch.arange(shape[1]),
            indexing="ij"
        )

        grid_x = grid_x.float() + dx * alpha
        grid_y = grid_y.float() + dy * alpha

        grid_x = 2.0 * grid_x / (shape[1] - 1) - 1.0
        grid_y = 2.0 * grid_y / (shape[0] - 1) - 1.0

        grid = torch.stack(
            [grid_x, grid_y],
            dim=-1
        ).unsqueeze(0)

        image = F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )[0]

        mask = F.grid_sample(
            mask.unsqueeze(0).float(),
            grid,
            mode='nearest',
            padding_mode='border',
            align_corners=False
        )[0]

        return image, mask

    def __getitem__(self, idx):

        image = self.images[idx]
        mask = self.masks[idx]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()

        if image.dim() == 2:
            image = image.unsqueeze(0)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if image.max() > 1:
            image = image / 255.0

        if self.is_train:

            # ----------------------------------
            # Horizontal Flip
            # ----------------------------------
            RAN = 0.5
            if random.random() < RAN:
                image = torch.flip(image, [-1])
                mask = torch.flip(mask, [-1])

            # ----------------------------------
            # Vertical Flip
            # ----------------------------------
            if random.random() < RAN:
                image = torch.flip(image, [-2])
                mask = torch.flip(mask, [-2])

            # ----------------------------------
            # Rotation
            # ----------------------------------
            if random.random() < RAN:

                angle = random.uniform(-15, 15)

                image = TF.rotate(
                    image,
                    angle,
                    interpolation=TF.InterpolationMode.BILINEAR
                )

                mask = TF.rotate(
                    mask,
                    angle,
                    interpolation=TF.InterpolationMode.NEAREST
                )

            # ----------------------------------
            # Affine
            # ----------------------------------
            # if random.random() < 0.8:

            #     angle = random.uniform(-10, 10)

            #     translate = (
            #         int(random.uniform(-0.1, 0.1) * image.shape[2]),
            #         int(random.uniform(-0.1, 0.1) * image.shape[1])
            #     )

            #     scale = random.uniform(0.8, 1.2)

            #     image = TF.affine(
            #         image,
            #         angle=angle,
            #         translate=translate,
            #         scale=scale,
            #         shear=0,
            #         interpolation=TF.InterpolationMode.BILINEAR
            #     )

            #     mask = TF.affine(
            #         mask,
            #         angle=angle,
            #         translate=translate,
            #         scale=scale,
            #         shear=0,
            #         interpolation=TF.InterpolationMode.NEAREST
            #     )

            # ----------------------------------
            # Random Crop + Resize
            # ----------------------------------
            if random.random() < RAN:

                H, W = image.shape[1:]

                crop_ratio = random.uniform(0.7, 1.0)

                crop_h = int(H * crop_ratio)
                crop_w = int(W * crop_ratio)

                top = random.randint(0, H - crop_h)
                left = random.randint(0, W - crop_w)

                image = TF.crop(
                    image,
                    top,
                    left,
                    crop_h,
                    crop_w
                )

                mask = TF.crop(
                    mask,
                    top,
                    left,
                    crop_h,
                    crop_w
                )

                image = TF.resize(
                    image,
                    [H, W],
                    interpolation=TF.InterpolationMode.BILINEAR
                )

                mask = TF.resize(
                    mask,
                    [H, W],
                    interpolation=TF.InterpolationMode.NEAREST
                )

            # ----------------------------------
            # Elastic
            # ----------------------------------
            # if random.random() < 0.01:
            #     image, mask = self._elastic_transform(
            #         image,
            #         mask,
            #         alpha=20,
            #         sigma=8
            #     )

            # ----------------------------------
            # Mild photometric augmentation
            # ----------------------------------
            if random.random() < 0.5:
                image = TF.adjust_brightness(
                    image,
                    random.uniform(0.85, 1.15)
                )

                image = TF.adjust_contrast(
                    image,
                    random.uniform(0.85, 1.15)
                )

                image = TF.adjust_saturation(
                    image,
                    random.uniform(0.9, 1.1)
                )

            # ----------------------------------
            # Mild blur/noise
            # ----------------------------------
            if random.random() < 0.15:
                image = TF.gaussian_blur(
                    image,
                    kernel_size=3
                )

            if random.random() < 0.25:
                noise = torch.randn_like(image) * 0.015
                image = image + noise

            #     image = image + noise

            # # ----------------------------------
            # # Cutout
            # # ----------------------------------
            # if random.random() < 0.9:

            #     H, W = image.shape[1:]

            #     size = random.randint(
            #         int(0.05 * H),
            #         int(0.15 * H)
            #     )

            #     y = random.randint(
            #         0,
            #         H - size
            #     )

            #     x = random.randint(
            #         0,
            #         W - size
            #     )

            #     image[:, y:y+size, x:x+size] = 0

        image = torch.clamp(image, 0, 1)

        mask = (mask > 0.5).float()

        return image, mask
   
class GradualWarmupScheduler:
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.optimizer = optimizer
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.images)
    def step(self, epoch):
        if epoch < self.total_epoch:
            progress = epoch / self.total_epoch
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * ((1 - progress) / self.multiplier + progress)
        else:
            if not self.finished:
                self.finished = True
            if self.after_scheduler:
                self.after_scheduler.step(epoch - self.total_epoch)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # pred: logits, target: binary [0,1]
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """Tversky loss for imbalanced segmentation."""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # Weight for False Positives
        self.beta = beta    # Weight for False Negatives (focus on polyps)
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        tp = (pred * target).sum()
        fp = ((1-target) * pred).sum()
        fn = (target * (1-pred)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        return 1 - tversky

class CombinedTverskyFocalLoss(nn.Module):
    """Combined Tversky and Focal loss for best performance."""
    def __init__(self, tversky_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight

    def forward(self, pred, target):
        return (self.tversky_weight * self.tversky(pred, target) + 
                self.focal_weight * self.focal(pred, target))
    
class BoundaryLoss(nn.Module):
    """Boundary-aware loss to improve segmentation edges"""
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
        
    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        
        # Target boundary detection
        target_boundary = F.max_pool2d(
            1 - target, kernel_size=self.theta0, stride=1, padding=self.theta0//2
        ) - (1 - target)
        
        # Distance-weighted cross entropy for boundaries
        dist_map = F.max_pool2d(
            target_boundary, kernel_size=self.theta, stride=1, padding=self.theta//2
        )
        dist_map = dist_map / (dist_map.max() + 1e-8)
        
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_bce = (bce * (1 + dist_map)).mean()
        
        return weighted_bce

class HybridLoss(nn.Module):
    """Combined loss for best segmentation performance"""
    def __init__(self, boundary_weight=0.3, tversky_weight=0.4, focal_weight=0.3):
        super().__init__()
        self.boundary = BoundaryLoss()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)
        self.boundary_weight = boundary_weight
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        
    def forward(self, pred, target):
        # Add label smoothing to prevent overfitting
        target_smooth = target * 0.9 + 0.05
        
        boundary_loss = self.boundary(pred, target_smooth)
        tversky_loss = self.tversky(pred, target_smooth)
        focal_loss = self.focal(pred, target_smooth)
        
        return (self.boundary_weight * boundary_loss + 
                self.tversky_weight * tversky_loss + 
                self.focal_weight * focal_loss)
    


def mixup_segmentation(data, target, alpha=0.2):
        """Mixup for segmentation"""
        batch_size = data.size(0)
        index = torch.randperm(batch_size).to(data.device)
        
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)  # Ensure lam >= 0.5
        
        mixed_data = lam * data + (1 - lam) * data[index]
        mixed_target = lam * target + (1 - lam) * target[index]
        
        return mixed_data, mixed_target

class SimpleCombinedLoss(nn.Module):
    """Dice + BCE with label smoothing"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5, label_smoothing=0.1):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.label_smoothing = label_smoothing
        
    def forward(self, pred, target):
        # Apply label smoothing
        target_smooth = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Dice loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target_smooth).sum()
        dice = 1 - (2. * intersection + 1e-6) / (pred_sigmoid.sum() + target_smooth.sum() + 1e-6)
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(pred, target_smooth)
        
        return self.dice_weight * dice + self.bce_weight * bce
    


# Replace SimpleCombinedLoss with this:
class StandardDiceBCELoss(nn.Module):
    """Dice + BCE WITHOUT label smoothing"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        
    def forward(self, pred, target):
        # NO label smoothing! Use hard 0 and 1 targets.
        bce = F.binary_cross_entropy_with_logits(pred, target)
        
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice = 1 - (2. * intersection + self.smooth) / (pred_sigmoid.sum() + target.sum() + self.smooth)
        
        return self.dice_weight * dice + self.bce_weight * bce

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )

        return 1 - dice.mean()

bce_loss_fn = nn.BCEWithLogitsLoss()






class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32)

        sobel_y = sobel_x.t()

        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        # 🔥 FORCE SAME DEVICE (IMPORTANT FIX)
        sobel_x = self.sobel_x.to(probs.device)
        sobel_y = self.sobel_y.to(probs.device)

        pred_edge = F.conv2d(probs, sobel_x, padding=1) + \
                    F.conv2d(probs, sobel_y, padding=1)

        target_edge = F.conv2d(targets, sobel_x, padding=1) + \
                      F.conv2d(targets, sobel_y, padding=1)

        return F.l1_loss(pred_edge, target_edge)
    

class CombinedLoss(nn.Module):
    def __init__(self, dice_w=0.4, bce_w=0.3, boundary_w=0.35):
        super().__init__()

        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.boundary = BoundaryLoss()

        self.dice_w = dice_w
        self.bce_w = bce_w
        self.boundary_w = boundary_w

    def forward(self, logits, targets):
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)
        boundary_loss = self.boundary(logits, targets)

        total = (
            self.dice_w * dice_loss +
            self.bce_w * bce_loss +
            self.boundary_w * boundary_loss
        )

        return total
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha   # FN weight
        self.beta = beta     # FP weight
        self.gamma = gamma   # focusing
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1 - targets)).sum(dim=1)
        fn = ((1 - probs) * targets).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )

        loss = (1 - tversky) ** self.gamma
        return loss.mean()



import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal

class DiceLossWithLogits(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1 - dice.mean()

class L1MaskLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        return F.l1_loss(probs, targets.float())

class CombinedSegLoss(nn.Module):
    def __init__(
        self,
        dice_w=0.35,
        bce_w=0.15,
        focal_w=0.15,
        smooth=1e-6,
        boundary_w= 0.35
    ):
        super().__init__()
        self.dice_w = dice_w
        self.bce_w = bce_w
        self.boundary_w = boundary_w
        self.focal_w = focal_w
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLossWithLogits(alpha=0.25, gamma=2.0)
    def boundary_loss(self, logits, targets):
        """Extra weight on boundary pixels using Sobel"""
        probs = torch.sigmoid(logits)
        targets = targets.float()
        
        # Sobel kernels
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], 
                        dtype=torch.float32, device=logits.device)
        kx = kx.view(1,1,3,3)
        ky = kx.transpose(-1,-2)
        
        # Get boundary map from GT
        if targets.dim() == 3:
            targets_4d = targets.unsqueeze(1)
        else:
            targets_4d = targets
        
        gx = F.conv2d(targets_4d, kx, padding=1)
        gy = F.conv2d(targets_4d, ky, padding=1)
        boundary = (torch.sqrt(gx**2 + gy**2) > 0.1).float()
        
        # Dilate boundary mask slightly
        boundary = F.max_pool2d(boundary, kernel_size=3, stride=1, padding=1)
        
        # Weighted BCE: boundary pixels count 3x
        weight = 1.0 + 2.0 * boundary
        bce_fn = nn.BCEWithLogitsLoss(reduction='none')
        return (bce_fn(logits, targets.float()) * weight).mean()
    
    def dice_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        
        # Flatten spatial dims
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()

    def forward(self, logits, targets):
        dice  = self.dice_loss(logits, targets)
        bce   = self.bce(logits, targets.float())
        focal = self.focal(logits, targets)
        boundary = self.boundary_loss(logits, targets)
        return (self.dice_w * dice + self.bce_w * bce + 
                self.focal_w * focal + self.boundary_w * boundary)
    


import torch
import torchvision.transforms.functional as TF


def tta_predict(model, image):
    """
    TTA using:
        - Original
        - Horizontal Flip
        - Vertical Flip
        - +10 Rotation
        - -10 Rotation

    Returns averaged probabilities.
    """

    model.eval()

    with torch.no_grad():

        probs = []

        # -----------------------------------
        # Original
        # -----------------------------------
        pred = torch.sigmoid(
            model(image)
        )

        probs.append(pred)

        # -----------------------------------
        # Horizontal Flip
        # -----------------------------------
        img_h = torch.flip(
            image,
            dims=[-1]
        )

        pred_h = torch.sigmoid(
            model(img_h)
        )

        pred_h = torch.flip(
            pred_h,
            dims=[-1]
        )

        probs.append(pred_h)

        # -----------------------------------
        # Vertical Flip
        # -----------------------------------
        img_v = torch.flip(
            image,
            dims=[-2]
        )

        pred_v = torch.sigmoid(
            model(img_v)
        )

        pred_v = torch.flip(
            pred_v,
            dims=[-2]
        )

        probs.append(pred_v)

        # -----------------------------------
        # Rotation +10°
        # -----------------------------------
        img_r1 = TF.rotate(
            image,
            angle=10,
            interpolation=TF.InterpolationMode.BILINEAR
        )

        pred_r1 = torch.sigmoid(
            model(img_r1)
        )

        pred_r1 = TF.rotate(
            pred_r1,
            angle=-10,
            interpolation=TF.InterpolationMode.BILINEAR
        )

        probs.append(pred_r1)

        # -----------------------------------
        # Rotation -10°
        # -----------------------------------
        img_r2 = TF.rotate(
            image,
            angle=-10,
            interpolation=TF.InterpolationMode.BILINEAR
        )

        pred_r2 = torch.sigmoid(
            model(img_r2)
        )

        pred_r2 = TF.rotate(
            pred_r2,
            angle=10,
            interpolation=TF.InterpolationMode.BILINEAR
        )

        probs.append(pred_r2)

        # -----------------------------------
        # Average probabilities
        # -----------------------------------
        prob_avg = torch.stack(
            probs,
            dim=0
        ).mean(dim=0)

        return prob_avg

import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_boundary_weights(mask, kappa=10.0):
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                             dtype=torch.float32, device=mask.device).view(1,1,3,3)
    sobel_y = sobel_x.transpose(2, 3)

    gx = F.conv2d(mask, sobel_x, padding=1)
    gy = F.conv2d(mask, sobel_y, padding=1)
    grad = (gx**2 + gy**2).sqrt()

    B = grad.shape[0]
    g_max = grad.view(B,-1).max(dim=1)[0].view(B,1,1,1).clamp(min=1e-6)
    alpha = grad / g_max

    return 1.0 + kappa * alpha   # w_ij


def weighted_bce(pred, gt, weights, eps=1e-6):
    pred = pred.sigmoid()
    loss = -(gt * torch.log(pred + eps) + (1 - gt) * torch.log(1 - pred + eps))
    return (weights * loss).sum() / (weights.sum() + eps)


def weighted_iou(pred, gt, weights, eps=1e-6):
    pred = pred.sigmoid()
    inter = (weights * gt * pred).sum(dim=(1,2,3))
    union = (weights * (gt + pred - gt * pred)).sum(dim=(1,2,3))
    return (1 - (inter + eps) / (union + eps)).mean()


class BoundaryAwareLoss(nn.Module):
    def __init__(self, kappa=10.0):
        super().__init__()
        self.kappa = kappa

    def forward(self, pred, gt):
        """
        pred : logits tensor (B,1,H,W)  OR  list of logits for deep supervision
        gt   : binary mask  (B,1,H,W)
        """
        gt = gt.float()

        if isinstance(pred, (list, tuple)):
            n = len(pred)
            stage_weights = [2**i for i in range(n)]   # higher-res → bigger weight
            total_w = sum(stage_weights)
            total_loss = 0.0
            for p, w in zip(pred, stage_weights):
                gt_r = F.interpolate(gt, size=p.shape[2:], mode='nearest')
                bw   = compute_boundary_weights(gt_r, self.kappa)
                total_loss += (w / total_w) * (weighted_bce(p, gt_r, bw) +
                                               weighted_iou(p, gt_r, bw))
            return total_loss

        bw = compute_boundary_weights(gt, self.kappa)
        return weighted_bce(pred, gt, bw) + weighted_iou(pred, gt, bw)

import torch
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probs.sum(dim=1) +
            targets.sum(dim=1) +
            self.smooth
        )

        return 1.0 - dice.mean()
    

class BoundaryDiceLoss(nn.Module):
    def __init__(
        self,
        kappa=10,
        boundary_weight=0.6,
        dice_weight=0.4,
    ):
        super().__init__()

        self.boundary = BoundaryAwareLoss(kappa=kappa)
        self.dice = SoftDiceLoss()

        self.boundary_weight = boundary_weight
        self.dice_weight = dice_weight

    def forward(self, logits, masks):

        loss_boundary = self.boundary(logits, masks)
        loss_dice = self.dice(logits, masks)

        loss = (
            self.boundary_weight * loss_boundary +
            self.dice_weight * loss_dice
        )

        return loss   
# Main execution
if __name__ == "__main__":
    # Create model
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
    )
    logging.info("#### Creating the model ####")
    model = None
    start_epoch = 0


        
    start_time = time.time()

    # Set default dtype to float32 (equivalent to TF's float32)
    # torch.set_default_dtype(torch.float32)
# jafari
    strtobool = (lambda s: s=='True')
    parser = argparse.ArgumentParser(description='TTFS')
    # Ablation reminder: keep all ConvNeXtUNet ablation variants on the shared
    # protocol in ABLATION_TRAINING_PROTOCOL.md. Do not change epochs, LR,
    # weight decay, loss, augmentation, scheduler, checkpoint selection, or
    # test-dataset usage for one variant unless the experiment explicitly says so.
    parser.add_argument('--data_name', type=str, default='KvasirSEG', help='(MNIST|CIFAR10|CIFAR100)')
    parser.add_argument('--logging_dir', type=str, default='./logs/ablation/baseline+MSC/', help='Directory for logging')
    parser.add_argument('--data_path', type=str, default='./data/', help='Directory for logging')
    parser.add_argument('--eval_dataset', type=str, default='both', choices=['kvasir', 'clinicdb', 'both', 'CVC-300', 'CVC-ColonDB', 'ETIS-LARIBPOLYPDB'], help='Validation/test split or external test dataset')
    parser.add_argument('--checkpoint_path', type=str, default='./logs/ablation/baseline+MSC/checkpoints_KvasirSEG-ConvNeXt/179-test0.86.pth', help='Checkpoint path used only when --load True')
    # parser.add_argument('--checkpoint_path', type=str, default='./logs/ConvNeXt-pretrain-lightweight/start-100/checkpoints_KvasirSEG-ConvNeXt/126-test0.88.pth', help='Checkpoint path used only when --load True')
    # parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint path used only when --load True')
    # parser.add_argument('--encoder_weights', type=str, default='./convnext_tiny_22k_1k_384.pth', help='ConvNeXt-Tiny pretrained encoder weights')
    parser.add_argument('--model_type', type=str, default='Gelu', help='(SNN|ReLU|Gelu)')
    parser.add_argument('--model_name', type=str, default='ConvNeXt', help='Should contain (FC2|VGG[BN]): e.g. VGG_BN_test1')
    parser.add_argument('--lr', type=float, default=1e-4, help='Baseline decoder learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--escape_lr', type=float, default=5e-5, help='Learning rate for escape')
    parser.add_argument('--batch_size', type=int, default=15, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50000, help='Epochs. 0 -skip training')
    parser.add_argument('--input_size', type=tuple, default=(352, 352), help='Input size for the images')
    parser.add_argument('--warmup_epochs', type=int, default=12, help='Epochs. 0 -skip training')
    parser.add_argument('--detail_warmup_epochs', type=int, default=0, help='Baseline disables detail warmup')
    parser.add_argument('--decoder_warmup_epochs', type=int, default=24, help='Train baseline decoder while keeping encoder frozen for this many epochs')
    parser.add_argument('--unfreeze_plateau_patience', type=int, default=10, help='Validation-IoU plateau epochs before exposing the next encoder level')
    parser.add_argument('--fixed_unfreeze_epochs', type=str, default='16,46,76,106,136', help='Five cumulative fixed-unfreeze milestones')
    parser.add_argument('--lr_plateau_patience', type=int, default=5, help='Validation-IoU plateau epochs before reducing learning rates')
    parser.add_argument('--lr_plateau_factor', type=float, default=0.9, help='ReduceLROnPlateau multiplicative LR reduction factor')
    parser.add_argument('--lr_scheduler', choices=['plateau', 'cosine_warm_restarts', 'warmup_cosine'], default='plateau', help='Learning-rate scheduler; plateau preserves the standard training protocol')
    parser.add_argument('--cosine_t0', type=int, default=8, help='First CosineAnnealingWarmRestarts cycle length')
    parser.add_argument('--cosine_t_mult', type=int, default=2, help='CosineAnnealingWarmRestarts cycle-length multiplier')
    parser.add_argument('--focal_tversky_after_warmup', type=strtobool, default=True, help='Enable Focal Tversky loss after detail warmup')
    parser.add_argument('--focal_tversky_w', type=float, default=0.05, help='Focal Tversky loss weight after detail warmup')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Decoder weight decay')
    parser.add_argument('--encoder_weight_decay', type=float, default=-1, help='Encoder weight decay; negative preserves the legacy ratio')
    parser.add_argument('--new_layer_weight_decay', type=float, default=1e-3, help='Weight decay for final_refine')
    parser.add_argument('--optimizer_profile', choices=['standard', 'layerwise_convnext'], default='standard', help='Optimizer parameter grouping profile')
    parser.add_argument('--encoder_layer_decay', type=float, default=0.8, help='Deep-to-shallow ConvNeXt LR decay')
    parser.add_argument('--max_grad_norm', type=float, default=0, help='Fixed gradient clipping norm; non-positive preserves staged defaults')
    parser.add_argument('--early_stop_patience', type=int, default=40, help='Stop training after this many epochs without validation IoU improvement. 0 disables it')
    parser.add_argument('--early_stop_start_epoch', type=int, default=0, help='One-based epoch after which early stopping may trigger')
    parser.add_argument('--use_ema', type=strtobool, default=True, help='Evaluate and save an exponential moving average of model weights')
    parser.add_argument('--ema_decay', type=float, default=0.995, help='EMA decay for model weights')
    parser.add_argument('--testing', type=strtobool, default=True, help='Execute testing.')
    parser.add_argument('--tta_check', type=strtobool, default=True, help='Execute testing.')
    parser.add_argument('--training', type=strtobool, default=False, help='Execute training.')
    parser.add_argument('--load', type=strtobool, default=True, help='Load checkpoint before training.')
    parser.add_argument('--resume_optimizer', type=strtobool, default=False, help='Resume optimizer and scheduler states when compatible')
    parser.add_argument('--refinement_checkpoint_path', type=str, default=None, help='Checkpoint used to initialize a fresh, isolated refinement run')
    parser.add_argument('--resume_lr_override', type=strtobool, default=False, help='Replace resumed optimizer LRs with the configured refinement LRs')
    parser.add_argument('--reset_plateau_scheduler', type=strtobool, default=False, help='Keep optimizer moments but start a fresh ReduceLROnPlateau state')
    parser.add_argument('--refinement_force_all_trainable', type=strtobool, default=False, help='Keep the entire encoder trainable when initializing refinement')
    parser.add_argument('--allow_completed_resume', type=strtobool, default=False, help='Explicitly continue a run whose checkpoint is marked complete')
    parser.add_argument('--save', type=strtobool, default=False, help='Store after training.')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise std.dev.')
    parser.add_argument('--time_bits', type=int, default=0, help='number of bits to represent time. 0 -disabled')
    parser.add_argument('--weight_bits', type=int, default=0, help='number of bits to represent weights. 0 -disabled')
    parser.add_argument('--w_min', type=float, default=-1.0, help='w_min to use if weight_bits is enabled')
    parser.add_argument('--w_max', type=float, default=1.0, help='w_max to use if weight_bits is enabled')
    parser.add_argument('--latency_quantiles', type=float, default=0.0, help='Number of quantiles for t_max. 0 -disabled')
    add_ablation_arguments(parser)
    args = parser.parse_args()
    fixed_unfreeze_epochs = parse_fixed_unfreeze_epochs(args.fixed_unfreeze_epochs)
    if args.deep_supervision_schedule == "anneal":
        if args.deep_supervision_heads == 0:
            raise ValueError("Deep-supervision annealing requires auxiliary heads")
        if not (
            0 < args.deep_supervision_anneal_start
            < args.deep_supervision_anneal_end
            <= args.epochs
        ):
            raise ValueError(
                "Deep-supervision annealing requires 0 < start < end <= epochs"
            )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.model_name = args.data_name + '-' + args.model_name
    set_up_logging(args.logging_dir, args.model_name)  # Assuming this exists

    robustness_params = {
        'noise': args.noise,
        'time_bits': args.time_bits,
        'weight_bits': args.weight_bits,
        'w_min': args.w_min,
        'w_max': args.w_max,
        'latency_quantiles': args.latency_quantiles
    }

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data object (assuming Dataset class returns PyTorch tensors)
    data = Dataset(
        args.data_name,
        args.logging_dir,
        flatten='FC' in args.model_name,
        ttfs_convert='SNN' in args.model_type,
        ttfs_noise=args.noise,
        data_path= args.data_path,
        input_size= args.input_size,
        evaluate_dataset=args.eval_dataset
    )

    train_dataset = KvasirSEGDataset(data.x_train, data.y_train, is_train=True, target_size=args.input_size[0])  # Pass target size to dataset
    test_dataset = KvasirSEGDataset(data.x_test, data.y_test, is_train=False, target_size=args.input_size[0])  # Pass target size to dataset

    train_sampler_generator = torch.Generator()
    train_sampler_generator.manual_seed(args.seed)
    frequency_generator = torch.Generator()
    frequency_generator.manual_seed(args.seed + 30030)
    if args.sampling_mode == "lesion_size_weighted":
        sample_weights, foreground_ratios = lesion_size_sampling_weights(data.y_train)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=train_sampler_generator,
        )
        logging.info(
            "Lesion-size weighted sampling: "
            f"<2%={(foreground_ratios < 0.02).sum()}, "
            f"2-10%={((foreground_ratios >= 0.02) & (foreground_ratios < 0.10)).sum()}, "
            f">=10%={(foreground_ratios >= 0.10).sum()}"
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(
            train_dataset, generator=train_sampler_generator
        )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    best_acc = 0.0



    from models.convnext_unet import *




    if 'ConvNeXt' in args.model_name:
        if 'Kvasir' in args.data_name:
            layers2D = [
                64, 64, 'pool',
                128, 128, 'pool',
                256, 256, 256, 'pool',
                512, 512, 512, 'pool',
                512, 512, 512
            ]

            layers1D = [512, 512]

        else:
            layers2D = [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool',
                       512, 512, 512, 'pool', 512, 512, 512, 'pool']
            layers1D = [512]
        
        kernel_size = (3, 3)
        BN = 'BN' in args.model_name

        if 'Gelu' in args.model_type:
            # from models.convnext_attention_BFIM import *
            # model = ConvNeXtTinyUNetAttention(
            #             dims=(96, 192, 384, 768),
            #             depths=(2, 2, 4, 2),
              
            #             dropout=0.1,  # INCREASE from 0.1 to 0.3
            #             drop_path_rate=0.1,  # INCREASE from 0.2 to 0.3
            #         )
            from models.convnext_pretrain import *
            # model = ConvNeXtTinyUNetAttention(
            #     in_channels=3,
            #     num_classes=1,
            #     encoder_pretrained=True,
            #     decoder_dims=(96, 192, 384, 768),
            #     bottleneck_dim=768,
            #     drop_path_rate=0.1
            # )
            encoder_weights = args.encoder_weights if args.encoder_weights and os.path.exists(args.encoder_weights) else None
            if encoder_weights is None:
                logging.warning(
                    f"Encoder pretrained weights not found at {args.encoder_weights}; "
                    "training encoder from scratch."
                )
            else:
                logging.info(f"Loading ConvNeXt-Tiny ImageNet encoder weights from {encoder_weights}.")
            if args.experiment_name and args.experiment_name.startswith("one_seed_"):
                from one_seed_models import build_experiment_model
                experiment_config = get_experiment(args.experiment_name)
                if bool(args.enable_csaf) != experiment_config.enable_csaf:
                    raise ValueError(
                        "--enable_csaf does not match the registered experiment"
                    )
                if bool(args.enable_fafem) != experiment_config.enable_fafem:
                    raise ValueError(
                        "--enable_fafem does not match the registered experiment"
                    )
                for stage_index in (1, 2, 3):
                    argument_value = bool(getattr(args, f"fafem_stage{stage_index}"))
                    config_value = getattr(experiment_config, f"fafem_stage{stage_index}")
                    if argument_value != config_value:
                        raise ValueError(
                            f"--fafem_stage{stage_index} does not match the registered experiment"
                        )
                if (bool(args.enable_gated_skip_stage3) !=
                        bool(getattr(experiment_config, "enable_gated_skip_stage3", False))):
                    raise ValueError(
                        "--enable_gated_skip_stage3 does not match the registered experiment"
                    )
                if (bool(args.enable_cross_level_fusion) !=
                        experiment_config.enable_cross_level_fusion):
                    raise ValueError(
                        "--enable_cross_level_fusion does not match the registered experiment"
                    )
                if (args.cross_level_fusion_version !=
                        experiment_config.cross_level_fusion_version):
                    raise ValueError(
                        "--cross_level_fusion_version does not match the registered experiment"
                    )
                if (bool(args.enable_geometry_conv_stage3) !=
                        bool(getattr(experiment_config, "enable_geometry_conv_stage3", False))):
                    raise ValueError(
                        "--enable_geometry_conv_stage3 does not match the registered experiment"
                    )
                if args.decoder_highres_width != getattr(experiment_config, "decoder_highres_width", 96):
                    raise ValueError(
                        "--decoder_highres_width does not match the registered experiment"
                    )
                if (bool(args.enable_mscb_lite_stage3) !=
                        bool(getattr(experiment_config, "enable_mscb_lite_stage3", False))):
                    raise ValueError(
                        "--enable_mscb_lite_stage3 does not match the registered experiment"
                    )
                if (bool(args.enable_fg_mscb_lite_stage3) !=
                        bool(getattr(experiment_config, "enable_fg_mscb_lite_stage3", False))):
                    raise ValueError(
                        "--enable_fg_mscb_lite_stage3 does not match the registered experiment"
                    )
                if (bool(args.enable_residual_fg_mscb_lite_stage3) !=
                        bool(getattr(experiment_config, "enable_residual_fg_mscb_lite_stage3", False))):
                    raise ValueError(
                        "--enable_residual_fg_mscb_lite_stage3 does not match the registered experiment"
                    )
                if (args.residual_fg_mscb_guidance_init_std !=
                        getattr(experiment_config, "residual_fg_mscb_guidance_init_std", 1e-3)):
                    raise ValueError(
                        "--residual_fg_mscb_guidance_init_std does not match the registered experiment"
                    )
                if (bool(args.residual_fg_mscb_signed_strength) !=
                        bool(getattr(experiment_config, "residual_fg_mscb_signed_strength", False))):
                    raise ValueError(
                        "--residual_fg_mscb_signed_strength does not match the registered experiment"
                    )
                if (args.residual_fg_mscb_initial_strength !=
                        getattr(experiment_config, "residual_fg_mscb_initial_strength", 0.0)):
                    raise ValueError(
                        "--residual_fg_mscb_initial_strength does not match the registered experiment"
                    )
                if (bool(args.enable_deformable_residual_fg_mscb_lite_stage3) !=
                        bool(getattr(experiment_config, "enable_deformable_residual_fg_mscb_lite_stage3", False))):
                    raise ValueError(
                        "--enable_deformable_residual_fg_mscb_lite_stage3 does not match the registered experiment"
                    )
                if (bool(args.enable_residual_fg_mscb_all_skips) !=
                        bool(getattr(experiment_config, "enable_residual_fg_mscb_all_skips", False))):
                    raise ValueError(
                        "--enable_residual_fg_mscb_all_skips does not match the registered experiment"
                    )
                if (bool(args.enable_partial_deformable_residual_fg_mscb_lite_stage3) !=
                        bool(getattr(experiment_config, "enable_partial_deformable_residual_fg_mscb_lite_stage3", False))):
                    raise ValueError(
                        "--enable_partial_deformable_residual_fg_mscb_lite_stage3 does not match the registered experiment"
                    )
                if (bool(args.enable_f4_f3_context_guided_mscb_lite_stage3) !=
                        bool(getattr(experiment_config, "enable_f4_f3_context_guided_mscb_lite_stage3", False))):
                    raise ValueError(
                        "--enable_f4_f3_context_guided_mscb_lite_stage3 does not match the registered experiment"
                    )
                if (bool(args.enable_mixstyle_stage1_stage2) !=
                        bool(getattr(experiment_config, "enable_mixstyle_stage1_stage2", False))):
                    raise ValueError(
                        "--enable_mixstyle_stage1_stage2 does not match the registered experiment"
                    )
                if (bool(args.enable_mscb_lite_stage2) !=
                        bool(getattr(experiment_config, "enable_mscb_lite_stage2", False))):
                    raise ValueError(
                        "--enable_mscb_lite_stage2 does not match the registered experiment"
                    )
                if (bool(args.enable_mscb_lite_stage1) !=
                        bool(getattr(experiment_config, "enable_mscb_lite_stage1", False))):
                    raise ValueError(
                        "--enable_mscb_lite_stage1 does not match the registered experiment"
                    )
                if (bool(args.enable_lka_lite_stage3) !=
                        bool(getattr(experiment_config, "enable_lka_lite_stage3", False))):
                    raise ValueError(
                        "--enable_lka_lite_stage3 does not match the registered experiment"
                    )
                if args.csaf_version != experiment_config.csaf_version:
                    raise ValueError(
                        "--csaf_version does not match the registered experiment"
                    )
                if args.unfreeze_schedule != experiment_config.unfreeze_schedule:
                    raise ValueError(
                        "--unfreeze_schedule does not match the registered experiment"
                    )
                if (bool(args.enable_frequency_augmentation) !=
                        experiment_config.enable_frequency_augmentation):
                    raise ValueError(
                        "--enable_frequency_augmentation does not match the registered experiment"
                    )
                if (args.uncertainty_refinement_version !=
                        experiment_config.uncertainty_refinement_version):
                    raise ValueError(
                        "--uncertainty_refinement_version does not match the registered experiment"
                    )
                model = build_experiment_model(
                    experiment_config, encoder_weights
                )
            else:
                model = ConvNeXtUNet(
                    weights_path=encoder_weights,
                    drop_path_rate=0.25,
                    dropout_rate=0.2,
                    encoder_depth=[3, 3, 9, 3],
                    enable_msc=args.enable_msc,
                    skip_mode=args.skip_mode,
                    detail_channels=args.detail_channels,
                    enable_gdf=args.enable_gdf,
                    deep_supervision_heads=args.deep_supervision_heads,
                    detail_fusion_mode=args.detail_fusion_mode,
                    enable_fafem=args.enable_fafem,
                    fafem_stage1=args.fafem_stage1,
                    fafem_stage2=args.fafem_stage2,
                    fafem_stage3=args.fafem_stage3,
                    enable_gated_skip_stage3=args.enable_gated_skip_stage3,
                    enable_cross_level_fusion=args.enable_cross_level_fusion,
                    cross_level_fusion_version=args.cross_level_fusion_version,
                    enable_geometry_conv_stage3=args.enable_geometry_conv_stage3,
                    decoder_highres_width=args.decoder_highres_width,
                    enable_mscb_lite_stage3=args.enable_mscb_lite_stage3,
                    enable_fg_mscb_lite_stage3=args.enable_fg_mscb_lite_stage3,
                    enable_residual_fg_mscb_lite_stage3=args.enable_residual_fg_mscb_lite_stage3,
                    residual_fg_mscb_guidance_init_std=args.residual_fg_mscb_guidance_init_std,
                    residual_fg_mscb_signed_strength=args.residual_fg_mscb_signed_strength,
                    residual_fg_mscb_initial_strength=args.residual_fg_mscb_initial_strength,
                    enable_deformable_residual_fg_mscb_lite_stage3=args.enable_deformable_residual_fg_mscb_lite_stage3,
                    enable_residual_fg_mscb_all_skips=args.enable_residual_fg_mscb_all_skips,
                    enable_partial_deformable_residual_fg_mscb_lite_stage3=args.enable_partial_deformable_residual_fg_mscb_lite_stage3,
                    enable_f4_f3_context_guided_mscb_lite_stage3=args.enable_f4_f3_context_guided_mscb_lite_stage3,
                    enable_mixstyle_stage1_stage2=args.enable_mixstyle_stage1_stage2,
                    enable_mscb_lite_stage2=args.enable_mscb_lite_stage2,
                    enable_mscb_lite_stage1=args.enable_mscb_lite_stage1,
                    enable_lka_lite_stage3=args.enable_lka_lite_stage3,
                )

            print('loaded lightweight ConvNeXt-Tiny U-Net')
    



    model = model.to(device)
    total_params, trainable_params, encoder_params_count = count_parameters(model)
    decoder_params_count = total_params - encoder_params_count
    logging.info(
        f"Parameter count: total={total_params / 1e6:.2f}M, "
        f"encoder={encoder_params_count / 1e6:.2f}M, "
        f"decoder_head={decoder_params_count / 1e6:.2f}M, "
        f"trainable={trainable_params / 1e6:.2f}M"
    )
    # print(model)

    

    class DiceBCEBoundaryLoss(nn.Module):
        def __init__(self, dice_w=0.55, bce_w=0.25, boundary_w=0.20, focal_tversky_w=0.0, label_smoothing=0.02):
            super().__init__()
            self.dice_w = dice_w
            self.bce_w = bce_w
            self.boundary_w = boundary_w
            self.focal_tversky_w = focal_tversky_w
            self.label_smoothing = label_smoothing
            self.bce = nn.BCEWithLogitsLoss()
            self.boundary = BoundaryAwareLoss(kappa=5)
            self.focal_tversky = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)

        def dice_loss(self, logits, targets, smooth=1.0):
            probs = torch.sigmoid(logits)
            probs = probs.view(probs.size(0), -1)
            targets = targets.view(targets.size(0), -1)
            inter = (probs * targets).sum(dim=1)
            dice = (2 * inter + smooth) / (
                probs.sum(dim=1) + targets.sum(dim=1) + smooth
            )
            return 1 - dice.mean()

        def forward(self, logits, targets):
            logits = logits.float()
            targets = targets.float()
            bce_targets = targets.float()
            if self.label_smoothing > 0:
                bce_targets = bce_targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            loss = (
                self.dice_w * self.dice_loss(logits, targets) +
                self.bce_w * self.bce(logits, bce_targets) +
                self.boundary_w * self.boundary(logits, targets)
            )
            if self.focal_tversky_w > 0:
                loss = loss + self.focal_tversky_w * self.focal_tversky(logits, targets)
            return loss
    if 'Kvasir' in args.data_name:

        # criterion = BoundaryAwareLoss(kappa=10)
        if args.experiment_name and args.experiment_name.startswith("one_seed_"):
            criterion = DiceBCEBoundaryLoss(
                dice_w=0.55, bce_w=0.25, boundary_w=0.20,
                focal_tversky_w=0.0, label_smoothing=0.02,
            )
        else:
            criterion = DiceBCEBoundaryLoss(
                dice_w=0.3, bce_w=0.3, boundary_w=0.40,
                focal_tversky_w=0.01, label_smoothing=0.02,
            )
        # criterion = BoundaryDiceLoss(
        #         kappa=10,
        #         boundary_weight=0.3,
        #         dice_weight=0.7
        #     )
        # criterion = CombinedSegLoss()
        # criterion = FocalTverskyLoss()
        # jafari
    else:
        criterion = nn.CrossEntropyLoss()

    start_epoch = 0

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #         optimizer, T_0=50, T_mult=1, eta_min=1e-5
    #     )
    from torch.optim.lr_scheduler import CyclicLR
 

    if args.experiment_name and args.experiment_name.startswith("one_seed_"):
        from one_seed_models import cosine_schedule_epochs
        total_epochs_remaining = cosine_schedule_epochs(
            args.epochs, args.detail_warmup_epochs, args.decoder_warmup_epochs
        )
    else:
        total_epochs_remaining = 200




   
      

    
    for param in model.parameters():
        param.requires_grad = True

 
    decoder_lr = args.lr
    encoder_lr = args.lr * 0.1
    refine_lr = args.lr * 1.5
    eta_min = min(args.min_lr, encoder_lr, decoder_lr, refine_lr)
    encoder_weight_decay = (
        args.encoder_weight_decay
        if args.encoder_weight_decay >= 0
        else args.weight_decay * 0.2
    )
    optimizer_groups = build_optimizer_param_groups(
        model,
        decoder_lr=decoder_lr,
        encoder_lr=encoder_lr,
        refine_lr=refine_lr,
        decoder_weight_decay=args.weight_decay,
        encoder_weight_decay=encoder_weight_decay,
        refine_weight_decay=args.new_layer_weight_decay,
        profile=args.optimizer_profile,
        encoder_layer_decay=args.encoder_layer_decay,
    )
    configured_group_lrs = {
        group["name"]: group["lr"] for group in optimizer_groups
    }
    optimizer = torch.optim.AdamW(optimizer_groups, betas=(0.9, 0.999))

    if args.lr_scheduler == 'warmup_cosine':
        scheduler = create_warmup_cosine_scheduler(
            optimizer, args.warmup_epochs, args.epochs, args.min_lr
        )
        scheduler_type = 'warmup_cosine'
    elif args.lr_scheduler == 'cosine_warm_restarts':
        if args.cosine_t0 <= 0 or args.cosine_t_mult < 1:
            raise ValueError("Cosine warm-restart periods must satisfy T_0 > 0 and T_mult >= 1")
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.cosine_t0,
            T_mult=args.cosine_t_mult,
            eta_min=args.min_lr,
        )
        scheduler_type = 'cosine_warm_restarts'
    else:
        scheduler = create_plateau_scheduler(
            optimizer, min_lr=args.min_lr, patience=args.lr_plateau_patience,
            factor=args.lr_plateau_factor,
        )
        scheduler_type = 'reduce_on_plateau'
    fafem_modules = (
        ("Bottleneck", getattr(model, "fafem", None)),
        ("Stage 3", getattr(model, "fafem_stage3", None)),
        ("Stage 2", getattr(model, "fafem_stage2", None)),
        ("Stage 1", getattr(model, "fafem_stage1", None)),
    )
    if any(module is not None for _, module in fafem_modules):
        per_module_params = {
            name: (sum(parameter.numel() for parameter in module.parameters())
                   if module is not None else 0)
            for name, module in fafem_modules
        }
        fafem_params = sum(per_module_params.values())
        logging.info(f"Experiment name: {args.experiment_name}")
        logging.info(f"Seed: {args.seed}")
        for name, module in fafem_modules:
            logging.info(
                f"FAFEM {name}: {'ON' if module is not None else 'OFF'} | "
                f"parameters={per_module_params[name]:,}"
            )
        logging.info(f"FAFEM parameter count: {fafem_params:,}")
        logging.info(f"Total parameter count: {total_params:,}")
        logging.info(f"Output directory: {args.seed_dir or args.logging_dir}")
    cross_level_fusion = getattr(model, "cross_level_fusion", None)
    if cross_level_fusion is not None:
        cross_level_params = sum(
            parameter.numel() for parameter in cross_level_fusion.parameters()
        )
        logging.info("Cross-Level Fusion: ON")
        logging.info(
            f"Cross-Level Fusion version: {args.cross_level_fusion_version}"
        )
        logging.info("Skip inputs: Stage 1, Stage 2, Stage 3")
        logging.info("Cross-Level Fusion outputs: Refined Stage 1, Refined Stage 2, Refined Stage 3")
        logging.info(f"Cross-Level Fusion parameters: {cross_level_params:,}")
    geometry_module = getattr(model, "geometry_conv_stage3", None)
    if geometry_module is not None:
        geometry_params = sum(parameter.numel() for parameter in geometry_module.parameters())
        logging.info("Geometry Conv Stage3: ON")
        logging.info(f"Geometry Conv Stage3 parameters: {geometry_params:,}")
    fg_mscb_module = getattr(model, "fg_mscb_lite_stage3", None)
    if fg_mscb_module is not None:
        fg_mscb_params = sum(parameter.numel() for parameter in fg_mscb_module.parameters())
        guidance_params = sum(
            parameter.numel() for parameter in fg_mscb_module.guidance_mlp.parameters()
        )
        logging.info("Frequency-guided MSCB-lite Stage3: ON")
        logging.info(f"FG-MSCB-lite parameters: {fg_mscb_params:,}")
        logging.info(f"FG-MSCB guidance MLP parameters: {guidance_params:,}")
    residual_fg_mscb_module = getattr(model, "residual_fg_mscb_lite_stage3", None)
    if residual_fg_mscb_module is not None:
        residual_fg_mscb_params = sum(
            parameter.numel() for parameter in residual_fg_mscb_module.parameters()
        )
        logging.info("Residual frequency-guided MSCB-lite Stage3: ON")
        logging.info(f"Residual FG-MSCB-lite parameters: {residual_fg_mscb_params:,}")
        logging.info(
            "Residual FG-MSCB guidance strength: %.6f (bounded to [0, 0.5])",
            residual_fg_mscb_module.effective_guidance_strength().item(),
        )
    f4_f3_context_module = getattr(model, "f4_f3_context_guided_mscb_lite_stage3", None)
    if f4_f3_context_module is not None:
        logging.info("F4+F3 context-guided residual FG-MSCB-lite Stage3: ON")
        logging.info("Descriptor shapes: f4_frequency_descriptor=[B,1536], f3_local_descriptor=[B,384], combined_descriptor=[B,1920]")
        logging.info("F4+F3 guidance MLP: 1920 -> 48 -> 3")
        logging.info(
            "F4+F3 residual FG-MSCB guidance strength: %.6f (bounded to [0, 0.5])",
            f4_f3_context_module.effective_guidance_strength().item(),
        )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = create_grad_scaler(amp_enabled, device.type)
    if args.experiment_name and args.experiment_name.startswith("one_seed_"):
        expected_precision = get_experiment(args.experiment_name).training_precision
        if expected_precision == "amp_fp16" and not args.amp:
            raise ValueError(f"{args.experiment_name} requires --amp True")
    logging.info(
        f"Training precision: {'amp_fp16' if amp_enabled else 'fp32'} | "
        f"GradScaler enabled={scaler.is_enabled()}"
    )

    logging.info(
        f"Optimizer: AdamW | refine_lr={refine_lr:.2e}, "
        f"encoder_lr={encoder_lr:.2e}, "
        f"decoder_lr={decoder_lr:.2e}, eta_min={eta_min:.2e}, "
        f"weight_decay={args.weight_decay:.2e}, "
        f"encoder_weight_decay={encoder_weight_decay:.2e}, "
        f"new_layer_weight_decay={args.new_layer_weight_decay:.2e}, "
        f"optimizer_profile={args.optimizer_profile}, "
        f"scheduler={scheduler_type}("
        + (
            f"T_0={args.cosine_t0},T_mult={args.cosine_t_mult},eta_min={args.min_lr:.2e})"
            if scheduler_type == 'cosine_warm_restarts'
            else (
                f"warmup_epochs={args.warmup_epochs},eta_min={args.min_lr:.2e})"
                if scheduler_type == 'warmup_cosine'
                else f"mode=max,factor={scheduler.factor},patience={scheduler.patience})"
            )
        )
    )
    for group in optimizer.param_groups:
        logging.info(
            f"Optimizer group {group.get('name')}: "
            f"parameters={sum(parameter.numel() for parameter in group['params']):,}, "
            f"max_lr={configured_group_lrs[group.get('name')]:.6e}, "
            f"weight_decay={group['weight_decay']:.2e}"
        )
    logging.info(
        "Gradient clipping: "
        + (f"fixed max_grad_norm={args.max_grad_norm:.2f}"
           if args.max_grad_norm > 0 else "staged legacy policy")
    )
    logging.info(
        "Deep supervision: "
        f"heads={args.deep_supervision_heads}, "
        f"schedule={args.deep_supervision_schedule}, "
        f"anneal={args.deep_supervision_anneal_start}-"
        f"{args.deep_supervision_anneal_end} | "
        f"sampling={args.sampling_mode}"
    )
    logging.info(
        "Frequency augmentation: "
        f"{'ON' if args.enable_frequency_augmentation else 'OFF'} | "
        f"max_probability={args.frequency_max_probability:g}, "
        f"region={100 * args.frequency_region_min:g}-"
        f"{100 * args.frequency_region_max:g}%, "
        f"max_mix={args.frequency_max_mix:g}, "
        f"anneal={100 * args.frequency_constant_fraction:g}-"
        f"{100 * args.frequency_anneal_end_fraction:g}%"
    )
    logging.info(
        "Uncertainty refinement: "
        f"{args.uncertainty_refinement_version}"
    )
    uncertainty_module = getattr(model, "uncertainty_refinement", None)
    if uncertainty_module is not None:
        logging.info(
            "Uncertainty refinement parameters: "
            f"{sum(parameter.numel() for parameter in uncertainty_module.parameters()):,}"
        )


    # for name, param in model.named_parameters():
    #     if 'encoder'  in name:
    #         param.requires_grad = False


#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode='max',          # because we're monitoring Dice (higher = better)
#     factor=0.75,          # halve the LR when plateauing
#     patience=4,         # wait 50 epochs before reducing
#     min_lr=1e-7,         # don't let it go too small
#     verbose=True
# )
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR



    from torch.optim.lr_scheduler import LambdaLR

    def get_triangular_scheduler(optimizer, min_lr, max_lr, epochs_to_peak, total_epochs):
        def lr_lambda(epoch):
            cycle_length = total_epochs
            epoch_in_cycle = epoch % cycle_length
            
            if epoch_in_cycle <= epochs_to_peak:
                # Ascending phase
                return 1.0 + (max_lr/min_lr - 1.0) * (epoch_in_cycle / epochs_to_peak)
            else:
                # Descending phase
                progress = (epoch_in_cycle - epochs_to_peak) / (total_epochs - epochs_to_peak)
                return max_lr/min_lr - (max_lr/min_lr - 1.0) * progress
        
        return LambdaLR(optimizer, lr_lambda)


    # scheduler = get_triangular_scheduler(optimizer, min_lr=lr, max_lr=0.0001, epochs_to_peak=80, total_epochs=160)
        
    checkpoint = None
    resumed_epochs_without_improvement = 0
    resumed_encoder_stages_unfrozen = 0
    resumed_stage_plateau_epochs = 0
    resumed_best_epoch = -1
    elapsed_seconds_before_resume = 0.0
    restart_stage_schedule = False
    initializing_refinement = False
    new_arch_prefixes = (
        "context.", "detail_fusion.", "uncertainty_refinement."
    )
    checkpoint_dir = Path(args.logging_dir) / f"checkpoints_{args.model_name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    canonical_checkpoint_path = (
        Path(args.best_checkpoint_path) if args.best_checkpoint_path
        else checkpoint_dir / "best.pth"
    )
    if args.experiment_name and args.seed_dir:
        experiment_config = get_experiment(args.experiment_name)
        decision = prepare_checkpoint(
            Path(args.seed_dir), experiment_config, args.seed, args.epochs,
            Path(args.logging_dir) / f"{args.model_name}_log.txt",
            allow_completed_resume=args.allow_completed_resume,
        )
        if decision.action == "error":
            raise RuntimeError(decision.reason)
        if decision.checkpoint_path is not None:
            args.load = True
            args.checkpoint_path = str(decision.checkpoint_path)
            logging.info(f"Checkpoint decision: {decision.action} - {decision.reason}")
        else:
            if args.refinement_checkpoint_path:
                refinement_checkpoint = Path(args.refinement_checkpoint_path)
                if not refinement_checkpoint.is_file():
                    raise FileNotFoundError(
                        f"Refinement checkpoint not found: {refinement_checkpoint}"
                    )
                args.load = True
                args.checkpoint_path = str(refinement_checkpoint)
                initializing_refinement = True
                logging.info(
                    "Initializing isolated refinement run from checkpoint: "
                    f"{refinement_checkpoint}"
                )
            else:
                args.load = False
    elif args.auto_resume:
        resume_path = prepare_best_checkpoint(checkpoint_dir)
        if resume_path is not None:
            args.load = True
            args.checkpoint_path = str(resume_path)
            logging.info(f"Auto-resume selected checkpoint: {resume_path}")
        else:
            args.load = False
            logging.info("Auto-resume found no valid checkpoint; starting a new run.")
    if args.load:
            if not args.checkpoint_path:
                raise ValueError("--load True requires --checkpoint_path")
            checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
            if initializing_refinement:
                expected_architecture = get_experiment(args.experiment_name).to_dict()
                source_architecture = dict(checkpoint.get('architecture') or {})
                expected_architecture.pop('name', None)
                source_architecture.pop('name', None)
                if source_architecture != expected_architecture:
                    raise ValueError(
                        "Refinement checkpoint architecture does not match the target "
                        f"experiment: source={source_architecture}, "
                        f"target={expected_architecture}"
                    )
                if int(checkpoint.get('seed', args.seed)) != args.seed:
                    raise ValueError("Refinement checkpoint seed does not match --seed")
            
            # Load model and optimizer states

            # remove old fusion weights
            # keys_to_remove = []

            # state_dict = checkpoint["model_state_dict"]
            # for k in state_dict.keys():

            #     if "bsei" in k:
            #         keys_to_remove.append(k)


            # for k in keys_to_remove:
            #     del state_dict[k]


            # msg = model.load_state_dict(
            #     state_dict,
            #     strict=False
            # )
            state_key = (
                'raw_model_state_dict'
                if args.training and 'raw_model_state_dict' in checkpoint
                else 'model_state_dict'
            )
            checkpoint_state = checkpoint[state_key]
            logging.info(f"Loading model weights from checkpoint['{state_key}'].")
            model_state = model.state_dict()
            compatible_state = {}
            skipped_keys = []

            for key, value in checkpoint_state.items():
                if key in model_state and model_state[key].shape == value.shape:
                    compatible_state[key] = value
                else:
                    skipped_keys.append(key)

            load_msg = model.load_state_dict(compatible_state, strict=False)
            logging.info(
                f"Loaded {len(compatible_state)}/{len(model_state)} compatible "
                f"model tensors from checkpoint."
            )
            if skipped_keys:
                logging.info(
                    "Skipped checkpoint tensors because they are new or shape-mismatched: "
                    f"{skipped_keys[:20]}"
                )
            if load_msg.missing_keys:
                logging.info(
                    f"Freshly initialized model tensors: {load_msg.missing_keys[:20]}"
                )
            restart_stage_schedule = any(
                key.startswith(new_arch_prefixes)
                for key in load_msg.missing_keys
            )
            if restart_stage_schedule:
                logging.info(
                    "New architecture tensors were missing in the checkpoint; "
                    "staged warmup will restart from the resumed epoch."
                )
            # model.encoder._load_weights(weights_path="./convnext_tiny_22k_1k_384.pth")

            checkpoint_optimizer_profile = checkpoint.get(
                'optimizer_profile', 'standard'
            )
            if checkpoint_optimizer_profile != args.optimizer_profile:
                raise ValueError(
                    "Checkpoint optimizer profile does not match the run: "
                    f"{checkpoint_optimizer_profile} != {args.optimizer_profile}"
                )
            if args.resume_optimizer:
                try:
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if (
                        not (initializing_refinement and args.reset_plateau_scheduler)
                        and checkpoint.get('scheduler_type') == scheduler_type
                        and 'scheduler_state_dict' in checkpoint
                    ):
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        if scheduler_type == 'reduce_on_plateau':
                            scheduler.patience = args.lr_plateau_patience
                            scheduler.factor = args.lr_plateau_factor
                        logging.info(
                            f"{scheduler_type} state resumed from checkpoint."
                        )
                    elif not (initializing_refinement and args.reset_plateau_scheduler):
                        configured_lrs = {
                            "refine": refine_lr,
                            "encoder": encoder_lr,
                            "decoder": decoder_lr,
                        }
                        for param_group in optimizer.param_groups:
                            group_name = param_group.get("name")
                            if group_name in configured_lrs:
                                param_group["lr"] = configured_lrs[group_name]
                                param_group["initial_lr"] = configured_lrs[group_name]
                        logging.info(
                            "Previous scheduler type is incompatible; "
                            "optimizer moments retained and configured learning "
                            "rates restored."
                        )
                    if initializing_refinement and args.resume_lr_override:
                        configured_lrs = {
                            "refine": refine_lr,
                            "encoder": encoder_lr,
                            "decoder": decoder_lr,
                        }
                        for param_group in optimizer.param_groups:
                            group_name = param_group.get("name")
                            if group_name in configured_lrs:
                                param_group["lr"] = configured_lrs[group_name]
                                param_group["initial_lr"] = configured_lrs[group_name]
                        logging.info(
                            "Resumed optimizer moments with refinement LR override: "
                            f"refine={refine_lr:.2e}, encoder={encoder_lr:.2e}, "
                            f"decoder={decoder_lr:.2e}."
                        )
                    if initializing_refinement and args.reset_plateau_scheduler:
                        logging.info(
                            f"{scheduler_type} state reset for isolated refinement."
                        )
                    restore_scaler_state(checkpoint, scaler, amp_enabled)
                    logging.info("Optimizer/scheduler state resumed from checkpoint.")
                except (ValueError, KeyError, RuntimeError) as exc:
                    logging.warning(
                        "Could not resume optimizer/scheduler state; continuing "
                        f"with fresh states. Reason: {exc}"
                    )
            # with torch.no_grad():
            #     model.bsei1.alpha.fill_(0.5)
            #     model.bsei2.alpha.fill_(0.5)
            #     model.bsei3.alpha.fill_(0.5)
            #         logging.info(f"Reset {name} to 0.1")

            start_epoch = checkpoint['epoch'] + 1
            best_acc = max(
                float(checkpoint.get('best_acc', 0.0)),
                float(checkpoint.get('test_iou', 0.0))
            )
            resumed_epochs_without_improvement = int(
                checkpoint.get('epochs_without_improvement', 0)
            )
            resumed_encoder_stages_unfrozen = int(
                checkpoint.get('encoder_stages_unfrozen', 0)
            )
            if initializing_refinement and args.refinement_force_all_trainable:
                resumed_encoder_stages_unfrozen = 5
            resumed_stage_plateau_epochs = int(
                checkpoint.get('stage_plateau_epochs', 0)
            )
            resumed_best_epoch = int(
                checkpoint.get('best_epoch', checkpoint.get('epoch', -1))
            )
            elapsed_seconds_before_resume = float(
                checkpoint.get('training_time_seconds', 0.0)
            )
            if initializing_refinement:
                elapsed_seconds_before_resume = 0.0
            
            logging.info(f"Resumed from epoch {checkpoint['epoch']}")
            logging.info(f"Previous best IoU: {best_acc:.4f}")
     
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = args.escape_lr

    if checkpoint is not None:
        checkpoint_sampling_mode = checkpoint.get("sampling_mode", "uniform")
        if checkpoint_sampling_mode != args.sampling_mode:
            raise ValueError(
                "Checkpoint sampling mode does not match the run: "
                f"{checkpoint_sampling_mode} != {args.sampling_mode}"
            )
        checkpoint_ds_schedule = checkpoint.get(
            "deep_supervision_schedule", "constant"
        )
        if checkpoint_ds_schedule != args.deep_supervision_schedule:
            raise ValueError(
                "Checkpoint deep-supervision schedule does not match the run: "
                f"{checkpoint_ds_schedule} != {args.deep_supervision_schedule}"
            )
        sampler_state = checkpoint.get("train_sampler_generator_state")
        if sampler_state is not None:
            train_sampler_generator.set_state(sampler_state.cpu())
            logging.info("Training sampler generator state resumed from checkpoint.")
        checkpoint_frequency_enabled = checkpoint.get(
            "enable_frequency_augmentation", False
        )
        if checkpoint_frequency_enabled != bool(args.enable_frequency_augmentation):
            raise ValueError("Checkpoint frequency-augmentation mode does not match the run")
        frequency_config = {
            "max_probability": args.frequency_max_probability,
            "max_mix": args.frequency_max_mix,
            "region_min": args.frequency_region_min,
            "region_max": args.frequency_region_max,
            "constant_fraction": args.frequency_constant_fraction,
            "anneal_end_fraction": args.frequency_anneal_end_fraction,
        }
        checkpoint_frequency_config = checkpoint.get("frequency_augmentation_config")
        if checkpoint_frequency_config is not None and checkpoint_frequency_config != frequency_config:
            raise ValueError("Checkpoint frequency-augmentation configuration does not match the run")
        frequency_state = checkpoint.get("frequency_generator_state")
        if frequency_state is not None:
            frequency_generator.set_state(frequency_state.cpu())
            logging.info("Frequency augmentation generator state resumed from checkpoint.")
        restore_rng_state(checkpoint)
        logging.info("Python, NumPy, Torch and CUDA RNG states resumed from checkpoint.")

    ema = None
    if args.use_ema:
        ema = ModelEMA(model, decay=args.ema_decay)
        if checkpoint is not None and "ema_state_dict" in checkpoint:
            model_state = model.state_dict()
            ema_state = {
                key: value
                for key, value in checkpoint["ema_state_dict"].items()
                if key in model_state and model_state[key].shape == value.shape
            }
            ema.load_state_dict(ema_state)
            logging.info(
                f"Loaded {len(ema_state)} compatible EMA tensors from checkpoint."
            )
        logging.info(f"EMA enabled with decay={args.ema_decay}.")

    if initializing_refinement:
        initial_refinement_checkpoint = {
            'epoch': int(checkpoint['epoch']),
            'model_state_dict': scored_model_state_dict(model, ema),
            'raw_model_state_dict': snapshot_state_dict(model),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scheduler_type': scheduler_type,
            'best_acc': best_acc,
            'test_dice': float(checkpoint.get('test_dice', 0.0)),
            'test_iou': float(checkpoint.get('test_iou', best_acc)),
            'epochs_without_improvement': 0,
            'encoder_stages_unfrozen': resumed_encoder_stages_unfrozen,
            'stage_plateau_epochs': 0,
            'experiment_name': args.experiment_name,
            'seed': args.seed,
            'architecture': get_experiment(args.experiment_name).to_dict(),
            'best_epoch': int(checkpoint.get('best_epoch', checkpoint['epoch'])),
            'best_validation_metric': best_acc,
            'final_epoch': None,
            'training_time_seconds': 0.0,
            'training_complete': False,
            'completion_reason': None,
            'training_precision': 'amp_fp16' if amp_enabled else 'fp32',
            'refinement_source_checkpoint': str(args.refinement_checkpoint_path),
            'lr_scheduler': args.lr_scheduler,
            'cosine_t0': args.cosine_t0,
            'cosine_t_mult': args.cosine_t_mult,
            'optimizer_profile': args.optimizer_profile,
            'encoder_layer_decay': args.encoder_layer_decay,
            'optimizer_max_lrs': configured_group_lrs,
        }
        add_scaler_state(initial_refinement_checkpoint, scaler, amp_enabled)
        if ema is not None:
            initial_refinement_checkpoint['ema_state_dict'] = ema.state_dict()
        atomic_save_checkpoint(initial_refinement_checkpoint, canonical_checkpoint_path)
        logging.info(
            "Seeded isolated refinement output with the source best checkpoint."
        )

    detail_warmup_epochs = max(args.detail_warmup_epochs, 0)
    decoder_warmup_epochs = max(args.decoder_warmup_epochs, 0)
    stage_schedule_start_epoch = start_epoch if restart_stage_schedule else 0
    detail_warmup_end_epoch = stage_schedule_start_epoch + detail_warmup_epochs
    full_train_start_epoch = detail_warmup_end_epoch + decoder_warmup_epochs

    if args.training and args.epochs > 0:
        if start_epoch < detail_warmup_end_epoch:
            stage = "detail"
        elif start_epoch < full_train_start_epoch:
            stage = "decoder"
        else:
            stage = (
                encoder_stage_name(resumed_encoder_stages_unfrozen)
                if args.experiment_name
                and args.experiment_name.startswith("one_seed_")
                and args.unfreeze_schedule != "none"
                else "all"
            )

        trainable_params, total_params = set_training_stage(model, stage=stage)
        logging.info(
            f"Training stage '{stage}' starts at epoch {start_epoch + 1} "
            f"({trainable_params:,}/{total_params:,} parameters trainable)."
        )
        if start_epoch >= detail_warmup_end_epoch and args.focal_tversky_after_warmup and hasattr(criterion, "focal_tversky_w"):
            criterion.focal_tversky_w = args.focal_tversky_w
            logging.info(
                f"Focal Tversky enabled with weight {criterion.focal_tversky_w:.3f}."
            )

    # evaluate.py measures complexity on an isolated model copy. Profiling the
    # training instance adds THOP buffers to checkpoints and breaks strict load.
    model.eval()

    # # Training
    best_threshold = 0.45
    if  args.testing:
        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
        best_threshold, best_dice, best_iou = find_best_threshold(
                model,
                test_loader,
                criterion,
                device
            )
        test_loss, test_dice, test_iou = test_segmentation(model, test_loader, criterion, device ,threshold=best_threshold ,use_tta=True)
        logging.info(
                        f"First EvaluationTest Loss: {test_loss:.4f}, "
                        f"Test Dice: {test_dice:.4f}, "
                        f"Test IoU: {test_iou:.4f}")
        if args.tta_check:
            tta_dice, tta_iou = evaluate_with_tta(model, test_loader, device, threshold=best_threshold)
            print(f"TTA      → Dice: {tta_dice:.4f}, IoU: {tta_iou:.4f}")
            print(f"IMPROVEMENT: +{(tta_iou - test_iou)*100:.2f}% IoU")        
        if ema is not None:
            ema.restore(model)
        # result = evaluate_model(
        #     model,
        #     test_loader,
        #     criterion,
        #     device,
        #     use_tta=True
        # )

        # print(
        #     f"Threshold={result['threshold']:.2f} "
        #     f"Dice={result['dice']:.4f} "
        #     f"IoU={result['iou']:.4f}"
        # )
    
    
    if  args.epochs > 0:
        logging.info("#### Training ####")
        total_steps = len(train_loader) * args.epochs
        warmup_steps = len(train_loader) * args.warmup_epochs
            
        last_step = -1
        
        saved_lr = args.lr  # Default to args.lr if not resuming
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        

        # Log the actual starting LR
        refine_start_lr = optimizer_lr(optimizer, "refine")
        enc_start_lr = encoder_optimizer_lr(optimizer)
        dec_start_lr = optimizer_lr(optimizer, "decoder", fallback_idx=-1)
        logging.info(
            f"initial LR: refine={refine_start_lr:.6f}, "
            f"encoder={enc_start_lr:.6f}, "
            f"decoder={dec_start_lr:.6f}"
        )
            
        # Training loop
        best_dice = 0
        epochs_without_improvement = resumed_epochs_without_improvement
        encoder_stages_unfrozen = resumed_encoder_stages_unfrozen
        stage_plateau_epochs = resumed_stage_plateau_epochs
        best_epoch = resumed_best_epoch
        completion_reason = "max_epochs"
        last_epoch = start_epoch - 1
        last_train_loss = None
        last_test_loss = None
        last_test_dice = None
        last_test_iou = None

        FREEZE_EPOCHS = start_epoch + 200
        if args.training:
            gradual_unfreezing = bool(
                args.experiment_name
                and args.experiment_name.startswith("one_seed_")
                and args.unfreeze_schedule != "none"
            )
            fixed_unfreezing = (
                gradual_unfreezing and args.unfreeze_schedule == "fixed"
            )
            if fixed_unfreezing:
                logging.info(
                    "Encoder unfreeze schedule: fixed cumulative milestones "
                    f"{fixed_unfreeze_epochs}; LR plateau scheduling remains enabled."
                )

            for epoch in range(start_epoch, args.epochs):
                current_ds_weights = deep_supervision_weights_for_epoch(
                    epoch + 1,
                    args.deep_supervision_schedule,
                    args.deep_supervision_anneal_start,
                    args.deep_supervision_anneal_end,
                )
                logging.info(
                    f"Epoch {epoch + 1}: deep-supervision weights "
                    f"main={current_ds_weights[0]:.4f}, "
                    f"d2={current_ds_weights[1]:.4f}, "
                    f"d3={current_ds_weights[2]:.4f}"
                )
                frequency_probability = (
                    frequency_augmentation_probability(
                        epoch + 1, args.epochs,
                        maximum=args.frequency_max_probability,
                        constant_fraction=args.frequency_constant_fraction,
                        anneal_end_fraction=args.frequency_anneal_end_fraction,
                    )
                    if args.enable_frequency_augmentation else 0.0
                )
                logging.info(
                    f"Epoch {epoch + 1}: frequency augmentation "
                    f"probability={frequency_probability:.4f}"
                )
                if (
                    gradual_unfreezing
                    and (
                        (fixed_unfreezing and
                         fixed_encoder_stages_for_epoch(epoch + 1, fixed_unfreeze_epochs) >
                         encoder_stages_unfrozen)
                        or (not fixed_unfreezing and
                            epoch == full_train_start_epoch and
                            encoder_stages_unfrozen == 0)
                    )
                ):
                    encoder_stages_unfrozen = (
                        fixed_encoder_stages_for_epoch(epoch + 1, fixed_unfreeze_epochs)
                        if fixed_unfreezing else 1
                    )
                    stage_plateau_epochs = 0
                    next_stage = encoder_stage_name(encoder_stages_unfrozen)
                    trainable_params, total_params = set_training_stage(
                        model, stage=next_stage
                    )
                    logging.info(
                        f"Epoch {epoch + 1}: "
                        f"{'fixed unfreeze milestone reached' if fixed_unfreezing else 'decoder-only warmup finished'}; "
                        f"stage '{next_stage}' active "
                        f"({trainable_params:,}/{total_params:,} parameters trainable)."
                    )
                if detail_warmup_epochs > 0 and epoch == detail_warmup_end_epoch:
                    next_stage = "decoder" if decoder_warmup_epochs > 0 else "all"
                    trainable_params, total_params = set_training_stage(model, stage=next_stage)
                    logging.info(
                        f"Epoch {epoch + 1}: detail warmup finished; "
                        f"stage '{next_stage}' active "
                        f"({trainable_params:,}/{total_params:,} parameters trainable)."
                    )
                    if args.focal_tversky_after_warmup and hasattr(criterion, "focal_tversky_w"):
                        criterion.focal_tversky_w = args.focal_tversky_w
                        logging.info(
                            f"Focal Tversky enabled with weight {criterion.focal_tversky_w:.3f}."
                        )

                legacy_transition = (
                    not gradual_unfreezing and epoch == full_train_start_epoch
                )
                if decoder_warmup_epochs > 0 and legacy_transition:
                    next_stage = "all"
                    trainable_params, total_params = set_training_stage(
                        model, stage=next_stage
                    )
                    logging.info(
                        f"Epoch {epoch + 1}: training "
                        f"stage '{next_stage}' active "
                        f"({trainable_params:,}/{total_params:,} "
                        "parameters trainable)."
                    )

                # if epoch == FREEZE_EPOCHS:
                #     
                #     logging.info(f"Epoch {epoch}: Backbone unfrozen, all params training")
                if 'Kvasir' in args.data_name:
                    train_result = train_epoch_segmentation(
                        model, train_loader, optimizer, criterion, device, scheduler,
                        threshold=best_threshold, ema=ema, amp_enabled=amp_enabled,
                        scaler=scaler,
                        max_grad_norm=(
                            args.max_grad_norm if args.max_grad_norm > 0
                            else grad_clip_norm_for_epoch(
                                epoch, full_train_start_epoch
                            )
                        ),
                        deep_supervision_weights=current_ds_weights,
                        frequency_augmentation_probability_value=frequency_probability,
                        frequency_generator=frequency_generator,
                        frequency_max_mix=args.frequency_max_mix,
                        frequency_region_min=args.frequency_region_min,
                        frequency_region_max=args.frequency_region_max,
                        collect_fg_mscb_alpha=bool(
                            args.enable_fg_mscb_lite_stage3 or
                            args.enable_residual_fg_mscb_lite_stage3 or
                            args.enable_deformable_residual_fg_mscb_lite_stage3 or
                            args.enable_residual_fg_mscb_all_skips or
                            args.enable_partial_deformable_residual_fg_mscb_lite_stage3
                            or args.enable_f4_f3_context_guided_mscb_lite_stage3
                        ),
                    )
                    if (args.enable_fg_mscb_lite_stage3 or
                            args.enable_residual_fg_mscb_lite_stage3 or
                            args.enable_deformable_residual_fg_mscb_lite_stage3 or
                            args.enable_residual_fg_mscb_all_skips or
                            args.enable_partial_deformable_residual_fg_mscb_lite_stage3 or
                            args.enable_f4_f3_context_guided_mscb_lite_stage3):
                        train_loss, fg_alpha_statistics = train_result
                    else:
                        train_loss = train_result
                        fg_alpha_statistics = None
                    if ema is not None:
                        ema.store(model)
                        ema.copy_to(model)
                    test_loss, test_dice, test_iou = test_segmentation(
                        model, test_loader, criterion, device,
                        threshold=best_threshold, amp_enabled=amp_enabled,
                    )
                    if ema is not None:
                        ema.restore(model)
                    if scheduler_type in {'cosine_warm_restarts', 'warmup_cosine'}:
                        scheduler.step()
                    elif epoch >= full_train_start_epoch:
                        scheduler.step(test_iou)

                    logging.info(f"Epoch {epoch+1}/{args.epochs}: "
                                f"Train Loss: {train_loss:.4f}, "
                                f"Test Loss: {test_loss:.4f}, "
                                f"Test Dice: {test_dice:.4f}, "
                                f"Test IoU: {test_iou:.4f}, "
                                f"EMA: {ema is not None}")
                    last_epoch = epoch
                    last_train_loss = train_loss
                    last_test_loss = test_loss
                    last_test_dice = test_dice
                    last_test_iou = test_iou
                    

                    # Flush immediately for Kvasir
                    for handler in logging.root.handlers:
                        handler.flush()

                    test_acc  = test_iou

           
   
                for handler in logging.root.handlers:
                    handler.flush()

                # Save best model
                is_best = test_acc > best_acc
                stage_changed = False
                if (gradual_unfreezing and not fixed_unfreezing and
                        epoch >= full_train_start_epoch):
                    (
                        encoder_stages_unfrozen,
                        stage_plateau_epochs,
                        stage_changed,
                    ) = advance_encoder_unfreezing(
                        encoder_stages_unfrozen,
                        stage_plateau_epochs,
                        improved=is_best,
                        patience=args.unfreeze_plateau_patience,
                    )
                    if stage_changed:
                        next_stage = encoder_stage_name(encoder_stages_unfrozen)
                        trainable_params, total_params = set_training_stage(
                            model, stage=next_stage
                        )
                        logging.info(
                            f"Epoch {epoch + 1}: validation IoU did not improve "
                            f"for {args.unfreeze_plateau_patience} epochs; "
                            f"stage '{next_stage}' active "
                            f"({trainable_params:,}/{total_params:,} parameters trainable)."
                        )
                elapsed_seconds_total = elapsed_seconds_before_resume + (time.time() - start_time)
                if args.training_history_path:
                    history_row = {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "validation_dice": test_dice,
                        "validation_iou": test_iou,
                        "encoder_lr": encoder_optimizer_lr(optimizer),
                        "decoder_lr": optimizer_lr(optimizer, "decoder", fallback_idx=-1),
                        "elapsed_seconds": elapsed_seconds_total,
                        "is_best": is_best,
                    }
                    if fg_alpha_statistics is not None:
                        history_row.update({
                            f"fg_mscb_{name}": value
                            for name, value in fg_alpha_statistics.items()
                            if name != "sample_count"
                        })
                    append_history_row(args.training_history_path, history_row)
                if is_best:
                    best_acc = test_acc
                    best_epoch = epoch
                    checkpoint_dict = {
                            'epoch': epoch,
                            'model_state_dict': scored_model_state_dict(model, ema),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),  # Now saving full state!
                            'scheduler_type': scheduler_type,
                            'best_acc': best_acc,
                            'test_dice': test_dice,
                            'test_iou': test_iou,
                            'epochs_without_improvement': 0,
                            'encoder_stages_unfrozen': encoder_stages_unfrozen,
                            'stage_plateau_epochs': stage_plateau_epochs,
                            'experiment_name': args.experiment_name,
                            'seed': args.seed,
                            'architecture': (
                                get_experiment(args.experiment_name).to_dict()
                                if args.experiment_name else None
                            ),
                            'best_epoch': epoch,
                            'best_validation_metric': best_acc,
                            'final_epoch': None,
                            'training_time_seconds': elapsed_seconds_total,
                            'training_complete': False,
                            'completion_reason': None,
                            'training_precision': 'amp_fp16' if amp_enabled else 'fp32',
                            'refinement_source_checkpoint': args.refinement_checkpoint_path,
                            'lr_scheduler': args.lr_scheduler,
                            'cosine_t0': args.cosine_t0,
                            'cosine_t_mult': args.cosine_t_mult,
                            'optimizer_profile': args.optimizer_profile,
                            'encoder_layer_decay': args.encoder_layer_decay,
                            'optimizer_max_lrs': configured_group_lrs,
                            'deep_supervision_schedule': args.deep_supervision_schedule,
                            'deep_supervision_anneal_start': args.deep_supervision_anneal_start,
                            'deep_supervision_anneal_end': args.deep_supervision_anneal_end,
                            'sampling_mode': args.sampling_mode,
                            'train_sampler_generator_state': train_sampler_generator.get_state(),
                            'enable_frequency_augmentation': bool(args.enable_frequency_augmentation),
                            'frequency_augmentation_config': {
                                'max_probability': args.frequency_max_probability,
                                'max_mix': args.frequency_max_mix,
                                'region_min': args.frequency_region_min,
                                'region_max': args.frequency_region_max,
                                'constant_fraction': args.frequency_constant_fraction,
                                'anneal_end_fraction': args.frequency_anneal_end_fraction,
                            },
                            'frequency_generator_state': frequency_generator.get_state(),
                            'frequency_augmentation_probability': frequency_probability,
                            **snapshot_rng_state(),
                        }
                    add_scaler_state(checkpoint_dict, scaler, amp_enabled)
                    if ema is not None:
                        checkpoint_dict['ema_state_dict'] = ema.state_dict()
                        checkpoint_dict['raw_model_state_dict'] = snapshot_state_dict(model)
                
                    if args.best_checkpoint_path:
                        best_checkpoint_path = atomic_save_checkpoint(
                            checkpoint_dict, canonical_checkpoint_path
                        )
                    else:
                        best_checkpoint_path = atomic_save_best(checkpoint_dict, checkpoint_dir)
                    logging.info(f"New best model saved with accuracy: {best_acc:.2f}%")
                    logging.info(f"Best checkpoint: {best_checkpoint_path}")
                    if args.tta_check:
                        if ema is not None:
                            ema.store(model)
                            ema.copy_to(model)
                        tta_dice, tta_iou = evaluate_with_tta(model, test_loader, device, threshold=best_threshold)
                        if ema is not None:
                            ema.restore(model)
                        print(f"TTA      → Dice: {tta_dice:.4f}, IoU: {tta_iou:.4f}")
                        print(f"IMPROVEMENT: +{(tta_iou - test_iou)*100:.2f}% IoU")
                    # Flush after saving
                    for handler in logging.root.handlers:
                        handler.flush()
                    epochs_without_improvement = 0
                else:
                    early_stopping_ready = (
                        epoch >= full_train_start_epoch
                        and (epoch + 1) > args.early_stop_start_epoch
                        and (
                            not gradual_unfreezing
                            or encoder_stages_unfrozen >= 5
                        )
                    )
                    if early_stopping_ready:
                        epochs_without_improvement += 1
                    else:
                        epochs_without_improvement = 0
                    if (
                        args.early_stop_patience > 0
                        and early_stopping_ready
                        and epochs_without_improvement >= args.early_stop_patience
                    ):
                        logging.info(
                            f"Early stopping at epoch {epoch + 1}: "
                            f"no validation IoU improvement for "
                            f"{epochs_without_improvement} epochs."
                        )
                        completion_reason = "early_stopping"
                if args.experiment_name and args.seed_dir:
                    latest_model_state = snapshot_state_dict(model)
                    latest_checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': latest_model_state,
                        'raw_model_state_dict': latest_model_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scheduler_type': scheduler_type,
                        'best_acc': best_acc,
                        'test_dice': test_dice,
                        'test_iou': test_iou,
                        'epochs_without_improvement': epochs_without_improvement,
                        'encoder_stages_unfrozen': encoder_stages_unfrozen,
                        'stage_plateau_epochs': stage_plateau_epochs,
                        'experiment_name': args.experiment_name,
                        'seed': args.seed,
                        'architecture': get_experiment(args.experiment_name).to_dict(),
                        'best_epoch': best_epoch,
                        'best_validation_metric': best_acc,
                        'final_epoch': None,
                        'training_time_seconds': elapsed_seconds_total,
                        'training_complete': False,
                        'completion_reason': None,
                        'training_precision': 'amp_fp16' if amp_enabled else 'fp32',
                        'refinement_source_checkpoint': args.refinement_checkpoint_path,
                        'lr_scheduler': args.lr_scheduler,
                        'cosine_t0': args.cosine_t0,
                        'cosine_t_mult': args.cosine_t_mult,
                        'optimizer_profile': args.optimizer_profile,
                        'encoder_layer_decay': args.encoder_layer_decay,
                        'optimizer_max_lrs': configured_group_lrs,
                        'deep_supervision_schedule': args.deep_supervision_schedule,
                        'deep_supervision_anneal_start': args.deep_supervision_anneal_start,
                        'deep_supervision_anneal_end': args.deep_supervision_anneal_end,
                        'sampling_mode': args.sampling_mode,
                        'train_sampler_generator_state': train_sampler_generator.get_state(),
                        'enable_frequency_augmentation': bool(args.enable_frequency_augmentation),
                        'frequency_augmentation_config': {
                            'max_probability': args.frequency_max_probability,
                            'max_mix': args.frequency_max_mix,
                            'region_min': args.frequency_region_min,
                            'region_max': args.frequency_region_max,
                            'constant_fraction': args.frequency_constant_fraction,
                            'anneal_end_fraction': args.frequency_anneal_end_fraction,
                        },
                        'frequency_generator_state': frequency_generator.get_state(),
                        'frequency_augmentation_probability': frequency_probability,
                        **snapshot_rng_state(),
                    }
                    add_scaler_state(latest_checkpoint, scaler, amp_enabled)
                    if ema is not None:
                        latest_checkpoint['ema_state_dict'] = ema.state_dict()
                    atomic_save_checkpoint(
                        latest_checkpoint,
                        Path(args.seed_dir) / "latest_checkpoint.pth",
                    )
                if completion_reason == "early_stopping":
                    break

        if args.training and last_epoch >= start_epoch:
            if canonical_checkpoint_path.is_file():
                best_checkpoint = load_checkpoint_file(canonical_checkpoint_path)
                training_time_seconds = elapsed_seconds_before_resume + (time.time() - start_time)
                completed_checkpoint = mark_training_complete(
                    best_checkpoint, last_epoch, training_time_seconds, completion_reason
                )
                atomic_save_checkpoint(completed_checkpoint, canonical_checkpoint_path)
                if args.seed_dir:
                    (Path(args.seed_dir) / "latest_checkpoint.pth").unlink(
                        missing_ok=True
                    )
                if args.training_summary_path:
                    write_training_summary(args.training_summary_path, {
                        "experiment_name": args.experiment_name,
                        "seed": args.seed,
                        "best_epoch": int(completed_checkpoint.get("best_epoch", completed_checkpoint["epoch"])),
                        "best_validation_metric": float(completed_checkpoint.get("best_validation_metric", best_acc)),
                        "final_epoch": last_epoch,
                        "training_time_seconds": training_time_seconds,
                        "best_checkpoint_path": str(canonical_checkpoint_path.resolve()),
                        "training_complete": True,
                        "completion_reason": completion_reason,
                        "training_precision": 'amp_fp16' if amp_enabled else 'fp32',
                    })
            logging.info(f"Training stopped after epoch {last_epoch + 1}; best checkpoint retained.")
            for handler in logging.root.handlers:
                handler.flush()
    # Save model
    if args.save and 'ReLU' in args.model_type:
        logging.info("#### Saving ReLU model ####")
        torch.save(model.state_dict(), f"{args.logging_dir}/{args.model_name}_weights.pth")
    
    print(f'### Total elapsed time [s]: {time.time() - start_time:.2f}')
































