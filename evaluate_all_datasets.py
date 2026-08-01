import argparse
import csv
import glob
import importlib
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:
    train_test_split = None


DATASETS = (
    "kvasir",
    "clinicdb",
    "both",
    "CVC-300",
    "CVC-ColonDB",
    "ETIS-LARIBPOLYPDB",
)


class PolypSegEvalDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).float()
        mask = torch.from_numpy(self.masks[idx]).float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        return image, mask


def read_split(data_path, dataset_name, input_size):
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    image_dir = os.path.join(data_path, dataset_name, "images")
    mask_dir = os.path.join(data_path, dataset_name, "masks")

    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    image_files = sorted(image_files)

    images = []
    masks = []
    for image_path in image_files:
        filename = os.path.basename(image_path)
        mask_path = os.path.join(mask_dir, filename)
        if not os.path.exists(mask_path):
            continue

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, input_size, interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        mask = (mask > 127).astype(np.float32)

        images.append(image)
        masks.append(mask)

    if not images:
        raise FileNotFoundError(f"No image/mask pairs found for {dataset_name} under {data_path}")

    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)


def load_eval_arrays(data_path, eval_dataset, input_size):
    kvasir_images, kvasir_masks = read_split(data_path, "Kvasir-SEG", input_size)
    clinic_images, clinic_masks = read_split(data_path, "CVC-ClinicDB", input_size)

    kvasir_val_x, kvasir_val_y = validation_split(kvasir_images, kvasir_masks)
    clinic_val_x, clinic_val_y = validation_split(clinic_images, clinic_masks)

    if eval_dataset == "kvasir":
        return kvasir_val_x, kvasir_val_y
    if eval_dataset == "clinicdb":
        return clinic_val_x, clinic_val_y
    if eval_dataset == "both":
        return (
            np.concatenate([kvasir_val_x, clinic_val_x], axis=0),
            np.concatenate([kvasir_val_y, clinic_val_y], axis=0),
        )
    return read_split(data_path, eval_dataset, input_size)


def validation_split(images, masks):
    if train_test_split is not None:
        _, val_x, _, val_y = train_test_split(
            images,
            masks,
            test_size=0.1,
            random_state=42,
            shuffle=True,
        )
        return val_x, val_y

    rng = np.random.RandomState(42)
    indices = np.arange(len(images))
    rng.shuffle(indices)
    val_count = int(np.ceil(len(images) * 0.1))
    val_indices = indices[-val_count:]
    return images[val_indices], masks[val_indices]


def make_loader(data_path, eval_dataset, input_size, batch_size, num_workers):
    images, masks = load_eval_arrays(data_path, eval_dataset, input_size)
    loader = DataLoader(
        PolypSegEvalDataset(images, masks),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, len(images)


def dice_coefficient(pred, target, smooth=1e-6):
    pred = pred.view(pred.size(0), -1)
    target = target.float().view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    return dice


def iou_score(pred, target, smooth=1e-6):
    pred = pred.view(pred.size(0), -1)
    target = target.float().view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return (intersection + smooth) / (union + smooth)


def evaluate(model, loader, device, threshold, use_tta=False):
    model.eval()
    losses = []
    dice_scores = []
    iou_scores = []
    maes = []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc=f"Eval threshold={threshold:.2f}", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            probs = tta_predict(model, images) if use_tta else torch.sigmoid(logits)
            if probs.shape[-2:] != masks.shape[-2:]:
                probs = F.interpolate(probs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            preds = (probs > threshold).float()

            losses.append(F.binary_cross_entropy_with_logits(logits, masks.float()).item())
            dice_scores.extend(dice_coefficient(preds, masks).detach().cpu().tolist())
            iou_scores.extend(iou_score(preds, masks).detach().cpu().tolist())
            maes.extend(torch.abs(probs - masks.float()).mean(dim=(1, 2, 3)).detach().cpu().tolist())

    return {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(dice_scores)),
        "iou": float(np.mean(iou_scores)),
        "mae": float(np.mean(maes)),
    }


def tta_predict(model, images):
    probs = []

    logits = model(images)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    probs.append(torch.sigmoid(logits))

    flipped = torch.flip(images, dims=[-1])
    logits = model(flipped)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    probs.append(torch.flip(torch.sigmoid(logits), dims=[-1]))

    flipped = torch.flip(images, dims=[-2])
    logits = model(flipped)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    probs.append(torch.flip(torch.sigmoid(logits), dims=[-2]))

    return torch.stack(probs, dim=0).mean(dim=0)


def find_best_threshold(model, loader, device, thresholds):
    best = {"threshold": 0.5, "dice": -1.0, "iou": 0.0, "loss": 0.0, "mae": 0.0}
    for threshold in thresholds:
        metrics = evaluate(model, loader, device, float(threshold), use_tta=False)
        print(
            f"threshold={threshold:.2f} "
            f"dice={metrics['dice']:.4f} "
            f"iou={metrics['iou']:.4f}"
        )
        if metrics["dice"] > best["dice"]:
            best = {"threshold": float(threshold), **metrics}
    return best


def build_model(model_spec, weights_path, device):
    module_name, class_name = model_spec.split(":")
    model_cls = getattr(importlib.import_module(module_name), class_name)
    model = model_cls(weights_path=weights_path, num_classes=1, encoder_depth=[3, 3, 9, 3])
    return model.to(device)


def load_checkpoint(model, checkpoint_path, device, state_key):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if state_key == "auto":
        state_key = "model_state_dict" if "model_state_dict" in checkpoint else None
    state_dict = checkpoint[state_key] if state_key else checkpoint

    model_state = model.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped.append(key)

    load_msg = model.load_state_dict(compatible_state, strict=False)
    print(f"Loaded {len(compatible_state)}/{len(model_state)} tensors from {checkpoint_path}")
    if load_msg.missing_keys:
        print(f"Missing model keys: {len(load_msg.missing_keys)}")
    if skipped:
        print(f"Skipped checkpoint keys: {len(skipped)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one ConvNeXtUNet checkpoint on all polyp datasets.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--encoder_weights", type=str, default="./convnext_tiny_22k_1k_384.pth")
    parser.add_argument("--model", type=str, default="models.convnext_pretrain:ConvNeXtUNet")
    parser.add_argument("--state_key", type=str, default="auto", help="auto, model_state_dict, raw_model_state_dict, or another checkpoint key")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--input_size", type=int, default=352)
    parser.add_argument("--threshold", type=str, default="auto", help="auto or a fixed value such as 0.50")
    parser.add_argument("--threshold_dataset", type=str, default="both", choices=("kvasir", "clinicdb", "both"))
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--output_csv", type=str, default="./all_dataset_eval_results.csv")
    parser.add_argument("--output_json", type=str, default="./all_dataset_eval_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = (args.input_size, args.input_size)

    model = build_model(args.model, args.encoder_weights, device)
    load_checkpoint(model, args.checkpoint_path, device, args.state_key)

    if args.threshold == "auto":
        print(f"\nFinding threshold on validation dataset: {args.threshold_dataset}")
        threshold_loader, _ = make_loader(
            args.data_path,
            args.threshold_dataset,
            input_size,
            args.batch_size,
            args.num_workers,
        )
        threshold_result = find_best_threshold(model, threshold_loader, device, np.arange(0.4, 0.99, 0.05))
        threshold = threshold_result["threshold"]
        print(f"Selected threshold={threshold:.2f} from {args.threshold_dataset} validation Dice")
    else:
        threshold = float(args.threshold)
        print(f"Using fixed threshold={threshold:.2f}")

    results = []
    print("\nEvaluating all datasets")
    for dataset_name in DATASETS:
        loader, num_images = make_loader(
            args.data_path,
            dataset_name,
            input_size,
            args.batch_size,
            args.num_workers,
        )
        metrics = evaluate(model, loader, device, threshold, use_tta=args.use_tta)
        row = {
            "dataset": dataset_name,
            "num_images": num_images,
            "threshold": threshold,
            "loss": metrics["loss"],
            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "mae": metrics["mae"],
            "tta": args.use_tta,
        }
        results.append(row)
        print(
            f"{dataset_name:18s} n={num_images:4d} "
            f"Dice={row['dice']:.4f} IoU={row['iou']:.4f} "
            f"MAE={row['mae']:.4f} Loss={row['loss']:.4f}"
        )

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved CSV: {args.output_csv}")
    print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
