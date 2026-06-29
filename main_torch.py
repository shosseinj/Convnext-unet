import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse
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

from PIL import Image
from sklearn.model_selection import train_test_split
import glob
import cv2


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
        input_size=(256, 256)
    ):

        self.name = data_name
        self.flatten = flatten
        self.data_path = data_path
        self.noise = ttfs_noise
        self.input_size = input_size
        self.logging_dir = logging_dir


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
        

        # eveluate_dataset = 'clinicdb'
        # eveluate_dataset = 'CVC-300'
        # eveluate_dataset = 'CVC-ColonDB'
        # eveluate_dataset = 'ETIS-LARIBPOLYPDB'
        eveluate_dataset = 'both'
        # eveluate_dataset = 'kvasir'
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


def train_epoch_segmentation(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    scheduler,
    threshold
):
    import random
    import numpy as np
    import logging
    from tqdm import tqdm
    import torch
    import torch.nn.functional as F

    model.train()

    running_loss = 0.0
    running_dice = 0.0

    skipped_batches = 0

    # ----------------------------
    # Diagnostics
    # ----------------------------
    enc_grad_sum = 0.0
    dec_grad_sum = 0.0
    prob_mean_sum = 0.0
    prob_std_sum = 0.0
    mask_area_sum = 0.0

    valid_batches = 0

    def grad_norm(module):
        total = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total += p.grad.detach().norm(2).item() ** 2
        return total ** 0.5

    pbar = tqdm(train_loader, desc="Training")

    for batch_idx, (data, target) in enumerate(pbar):

        data = data.to(device)
        target = target.to(device)

        data = torch.clamp(data, 0.0, 1.0)

        if target.dim() == 3:
            target = target.unsqueeze(1)

        # MixUp
        if random.random() < 0.4:
            lam = np.random.beta(0.2, 0.2)
            idx = torch.randperm(data.size(0), device=device)

            data = lam * data + (1 - lam) * data[idx]
            target = lam * target + (1 - lam) * target[idx]

        optimizer.zero_grad()

        # ----------------------------
        # Forward
        # ----------------------------
        output = model(data)

        if not torch.isfinite(output).all():
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        if output.shape != target.shape:
            output = F.interpolate(
                output,
                size=target.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        loss = criterion(output, target)

        if not torch.isfinite(loss):
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        # ----------------------------
        # Backward
        # ----------------------------
        try:
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
            optimizer.zero_grad(set_to_none=True)
            continue

        # ----------------------------
        # Diagnostics BEFORE clipping
        # ----------------------------

        enc_grad = grad_norm(model.encoder)
        dec_grad = grad_norm_except_encoder(model)

        grad_norm_total = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.5
        )

        optimizer.step()

        # scheduler.step()

        with torch.no_grad():

            prob = torch.sigmoid(output)

            pred = (prob > threshold).float()

            dice = dice_coefficient(pred, target)

            prob_mean = prob.mean().item()
            prob_std = prob.std().item()
            mask_area = pred.mean().item()

        # ----------------------------
        # Accumulate
        # ----------------------------

        running_loss += loss.item()
        running_dice += dice

        enc_grad_sum += enc_grad
        dec_grad_sum += dec_grad

        prob_mean_sum += prob_mean
        prob_std_sum += prob_std
        mask_area_sum += mask_area

        valid_batches += 1

        # Current learning rates

        enc_lr = optimizer.param_groups[0]["lr"]

        dec_lr = (
            optimizer.param_groups[1]["lr"]
            if len(optimizer.param_groups) > 1
            else enc_lr
        )

        pbar.set_postfix(

            Loss=f"{loss.item():.4f}",

            Dice=f"{dice:.4f}",

            EncGrad=f"{enc_grad:.2f}",

            DecGrad=f"{dec_grad:.2f}",

            Prob=f"{prob_mean:.3f}",

            Area=f"{mask_area:.3f}",

            EncLR=f"{enc_lr:.2e}",

            DecLR=f"{dec_lr:.2e}",

            Skip=skipped_batches,
        )

    if skipped_batches > 0:
        logging.warning(
            f"Skipped {skipped_batches}/{len(train_loader)} batches."
        )

    # ----------------------------
    # Epoch summary
    # ----------------------------

    if valid_batches > 0:

        logging.info(
            "\n"
            f"Train Dice      : {running_dice / valid_batches:.4f}\n"
            f"Train Loss      : {running_loss / valid_batches:.4f}\n"
            f"Encoder Grad    : {enc_grad_sum / valid_batches:.4f}\n"
            f"Decoder Grad    : {dec_grad_sum / valid_batches:.4f}\n"
            f"Mean Prob       : {prob_mean_sum / valid_batches:.4f}\n"
            f"Prob Std        : {prob_std_sum / valid_batches:.4f}\n"
            f"Mask Area       : {mask_area_sum / valid_batches:.4f}\n"
            f"Encoder LR      : {optimizer.param_groups[0]['lr']:.2e}\n"
            f"Decoder LR      : {optimizer.param_groups[1]['lr']:.2e}"
        )

    return running_loss / max(valid_batches, 1)

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


def evaluate_with_tta(model, test_loader, device, threshold=0.40):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='TTA'):
            data = data.to(device)
            
            # Original
            pred1 = torch.sigmoid(model(data))
            
            # Horizontal flip
            pred2 = torch.flip(torch.sigmoid(model(torch.flip(data, [-1]))), [-1])
            
            # Vertical flip  
            pred3 = torch.flip(torch.sigmoid(model(torch.flip(data, [-2]))), [-2])
            
            # Average
            pred = (pred1 + pred2 + pred3) / 3
            
            all_preds.append((pred > threshold).float().cpu())
            all_targets.append(target.cpu())
    
    # Calculate metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    dice = dice_coefficient(all_preds, all_targets)
    iou = iou_score(all_preds, all_targets)
    
    return dice, iou


def test_segmentation(model, test_loader, criterion, device, threshold=0.3, use_tta=False):
    """Testing function with optional TTA"""
    model.eval()
    test_loss = 0
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
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
                output = model(data)
                if output.shape != target.shape:
                    output = F.interpolate(output, size=target.shape[2:], mode='bilinear', align_corners=False)
                test_loss += criterion(output, target).item()
            else:
                # Regular prediction
                output = model(data)
                if output.shape != target.shape:
                    output = F.interpolate(output, size=target.shape[2:], mode='bilinear', align_corners=False)
                test_loss += criterion(output, target).item()
                pred = (torch.sigmoid(output) > threshold).float()
            
            dice_scores.append(dice_coefficient(pred, target))
            iou_scores.append(iou_score(pred, target))
    
    avg_loss = test_loss / len(test_loader)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    
    return avg_loss, avg_dice, avg_iou

def find_best_threshold(model, test_loader, criterion, device):
    model.eval()

    thresholds = np.arange(0.4, 0.95, 0.05)
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

        for data, target in tqdm(loader, desc="Evaluating"):

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
            RAN = 0.6
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
            if random.random() < 0.8:

                angle = random.uniform(-10, 10)

                translate = (
                    int(random.uniform(-0.1, 0.1) * image.shape[2]),
                    int(random.uniform(-0.1, 0.1) * image.shape[1])
                )

                scale = random.uniform(0.8, 1.2)

                image = TF.affine(
                    image,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=0,
                    interpolation=TF.InterpolationMode.BILINEAR
                )

                mask = TF.affine(
                    mask,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=0,
                    interpolation=TF.InterpolationMode.NEAREST
                )

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
            if random.random() < 0.01:
                image, mask = self._elastic_transform(
                    image,
                    mask,
                    alpha=20,
                    sigma=8
                )

            # ----------------------------------
            # Brightness
            # ----------------------------------
            if random.random() < 0.9:

                image = TF.adjust_brightness(
                    image,
                    random.uniform(0.8, 1.2)
                )

                image = TF.adjust_contrast(
                    image,
                    random.uniform(0.8, 1.2)
                )

                image = TF.adjust_saturation(
                    image,
                    random.uniform(0.8, 1.2)
                )

            # ----------------------------------
            # Gaussian Blur
            # ----------------------------------
            if random.random() < 0.8:

                image = TF.gaussian_blur(
                    image,
                    kernel_size=5
                )

            # ----------------------------------
            # Gaussian Noise
            # ----------------------------------
            if random.random() < 0.8:

                noise = (
                    torch.randn_like(image) * 0.03
                )

                image = image + noise

            # ----------------------------------
            # Cutout
            # ----------------------------------
            if random.random() < 0.9:

                H, W = image.shape[1:]

                size = random.randint(
                    int(0.05 * H),
                    int(0.15 * H)
                )

                y = random.randint(
                    0,
                    H - size
                )

                x = random.randint(
                    0,
                    W - size
                )

                image[:, y:y+size, x:x+size] = 0

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
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("#### Creating the model ####")
    model = None
    start_epoch = 0


        
    start_time = time.time()

    # Set default dtype to float32 (equivalent to TF's float32)
    # torch.set_default_dtype(torch.float32)

    strtobool = (lambda s: s=='True')
    path_weight = './logs/ConvNeXt-pretrain/start-93.88/checkpoints_KvasirSEG-ConvNeXt/2617-test0.87.pth'
    parser = argparse.ArgumentParser(description='TTFS')
    parser.add_argument('--data_name', type=str, default='KvasirSEG', help='(MNIST|CIFAR10|CIFAR100)')
    parser.add_argument('--logging_dir', type=str, default='./logs/ConvNeXt-pretrain_depth2242/start/', help='Directory for logging')
    parser.add_argument('--data_path', type=str, default='./data/', help='Directory for logging')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Directory for logging')
    # parser.add_argument('--checkpoint_path', type=str, default=path_weight, help='Directory for logging')
    parser.add_argument('--model_type', type=str, default='Gelu', help='(SNN|ReLU|Gelu)')
    parser.add_argument('--model_name', type=str, default='ConvNeXt', help='Should contain (FC2|VGG[BN]): e.g. VGG_BN_test1')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--escape_lr', type=float, default=5e-5, help='Learning rate for escape')
    parser.add_argument('--batch_size', type=int, default=25, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50000, help='Epochs. 0 -skip training')
    parser.add_argument('--input_size', type=tuple, default=(352, 352), help='Input size for the images')
    parser.add_argument('--warmup_epochs', type=int, default=4, help='Epochs. 0 -skip training')
    parser.add_argument('--testing', type=strtobool, default=False, help='Execute testing.')
    parser.add_argument('--tta_check', type=strtobool, default=False, help='Execute testing.')
    parser.add_argument('--training', type=strtobool, default=True, help='Execute training.')
    parser.add_argument('--load', type=str, default=False, help='Load before training.')
    parser.add_argument('--save', type=strtobool, default=False, help='Store after training.')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise std.dev.')
    parser.add_argument('--time_bits', type=int, default=0, help='number of bits to represent time. 0 -disabled')
    parser.add_argument('--weight_bits', type=int, default=0, help='number of bits to represent weights. 0 -disabled')
    parser.add_argument('--w_min', type=float, default=-1.0, help='w_min to use if weight_bits is enabled')
    parser.add_argument('--w_max', type=float, default=1.0, help='w_max to use if weight_bits is enabled')
    parser.add_argument('--latency_quantiles', type=float, default=0.0, help='Number of quantiles for t_max. 0 -disabled')
    args = parser.parse_args()

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
        input_size= args.input_size
    )

    train_dataset = KvasirSEGDataset(data.x_train, data.y_train, is_train=True, target_size=args.input_size[0])  # Pass target size to dataset
    test_dataset = KvasirSEGDataset(data.x_test, data.y_test, is_train=False, target_size=args.input_size[0])  # Pass target size to dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
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
            model = ConvNeXtUNet(
        # weights_path="./convnext_tiny_22k_1k_384.pth",
                drop_path_rate=0.4,
                dropout_rate=0.4,
                encoder_depth= [2,2,4,2]
            )
            model.encoder._load_weights(weights_path="./convnext_tiny_22k_1k_384.pth")

            print('loaded Relu version of ConvNeXt-Tiny')
    



    model = model.to(device)
    # print(model)

    

    

    if 'Kvasir' in args.data_name:

        criterion = BoundaryAwareLoss(kappa=10)
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
 

    total_epochs_remaining = 200  # train for 200 more epochs then evaluate




   
      

    
    for param in model.parameters():
        param.requires_grad = True

 
    lr = 0.0001   
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    other_params = []
    for name, param in model.named_parameters():
        if 'encoder' not in name:  # or 'ConvNeXtEncoder' depending on your model
            other_params.append(param)


    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": 1e-5,
            },
            {
                "params": other_params,
                "lr": 4e-4,
            },
        ],
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=80,
        eta_min= 0.00006  
    )  


    for name, param in model.named_parameters():
        if 'encoder'  in name:
            param.requires_grad = False


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
        
    if args.checkpoint_path:
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            
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
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.encoder._load_weights(weights_path="./convnext_tiny_22k_1k_384.pth")

            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # DON'T load scheduler state - it was a different scheduler!
            # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # REMOVE THIS
            # with torch.no_grad():
            #     model.bsei1.alpha.fill_(0.5)
            #     model.bsei2.alpha.fill_(0.5)
            #     model.bsei3.alpha.fill_(0.5)
            #         logging.info(f"Reset {name} to 0.1")

            logging.info(f"Resumed from epoch {checkpoint['epoch']}, best_acc={best_acc:.4f}")


            start_epoch = checkpoint['epoch'] + 1
            # best_acc = 0
            best_acc = checkpoint['best_acc']
            
        
            
            logging.info(f"Resumed from epoch {checkpoint['epoch']}")
            logging.info(f"Previous best accuracy: {best_acc:.4f}")
     
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = args.escape_lr

    from torchinfo import summary
    summary(
            model,
            input_size=(1, 3) + args.input_size,  # (batch, channels, H, W)
            device="cuda"
        )
    
    # # Training
    best_threshold = 0.45
    if  args.testing:
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
        
        os.makedirs(f"{args.logging_dir}checkpoints_{args.model_name}", exist_ok=True)
        

        # Log the actual starting LR
        current_lr = optimizer.param_groups[0]['lr']
        logging.info( f"initial LR: {current_lr:.6f}")
            
        # Training loop
        best_dice = 0

        FREEZE_EPOCHS = start_epoch + 200
        if args.training:



            for epoch in range(start_epoch, args.epochs):
                # if epoch == FREEZE_EPOCHS:
                #     
                #     logging.info(f"Epoch {epoch}: Backbone unfrozen, all params training")
                if 'Kvasir' in args.data_name:
                    train_loss = train_epoch_segmentation(model, train_loader, optimizer, criterion, device,scheduler,threshold=best_threshold)
                    test_loss, test_dice, test_iou = test_segmentation(model, test_loader, criterion, device, threshold=best_threshold)
                    scheduler.step()

                    logging.info(f"Epoch {epoch+1}/{args.epochs}: "
                                f"Train Loss: {train_loss:.4f}, "
                                f"Test Loss: {test_loss:.4f}, "
                                f"Test Dice: {test_dice:.4f}, "
                                f"Test IoU: {test_iou:.4f}")
                    

                    # Flush immediately for Kvasir
                    for handler in logging.root.handlers:
                        handler.flush()
                    
                    test_acc  = test_iou

           
   
                for handler in logging.root.handlers:
                    handler.flush()

                # Save best model
                if test_acc > best_acc:
                    checkpoint_dict = {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),  # Now saving full state!
                            'best_acc': best_acc,
                            'test_dice': test_dice,
                        }
                
                    best_acc = test_acc
                    torch.save(checkpoint_dict, 
                            f"{args.logging_dir}checkpoints_{args.model_name}/{epoch}-test{test_acc:.2f}.pth")
                    logging.info(f"New best model saved with accuracy: {best_acc:.2f}%")
                    if args.tta_check:
                        tta_dice, tta_iou = evaluate_with_tta(model, test_loader, device, threshold=best_threshold)
                        print(f"TTA      → Dice: {tta_dice:.4f}, IoU: {tta_iou:.4f}")
                        print(f"IMPROVEMENT: +{(tta_iou - test_iou)*100:.2f}% IoU")
                    # Flush after saving
                    for handler in logging.root.handlers:
                        handler.flush()
    # Save model
    if args.save and 'ReLU' in args.model_type:
        logging.info("#### Saving ReLU model ####")
        torch.save(model.state_dict(), f"{args.logging_dir}/{args.model_name}_weights.pth")
    
    print(f'### Total elapsed time [s]: {time.time() - start_time:.2f}')
































