"""Shared dataset loading, segmentation metrics, and model complexity."""

import copy
import glob
import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset


DATASET_DIRECTORIES = {
    "Kvasir-SEG": "Kvasir-SEG",
    "CVC-ClinicDB": "CVC-ClinicDB",
    "CVC-300": "CVC-300",
    "CVC-ColonDB": "CVC-ColonDB",
    "ETIS-LaribPolypDB": "ETIS-LARIBPOLYPDB",
}


class PolypEvalDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return torch.from_numpy(self.images[index]).float(), torch.from_numpy(
            self.masks[index]
        ).float().unsqueeze(0)


def read_dataset(data_path, dataset_key, input_size=352):
    directory = DATASET_DIRECTORIES[dataset_key]
    image_dir = os.path.join(data_path, directory, "images")
    mask_dir = os.path.join(data_path, directory, "masks")
    image_files = []
    for extension in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        image_files.extend(glob.glob(os.path.join(image_dir, extension)))
    images, masks = [], []
    for image_path in sorted(image_files):
        mask_path = os.path.join(mask_dir, os.path.basename(image_path))
        if not os.path.isfile(mask_path):
            continue
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
        images.append(np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)))
        masks.append((mask > 127).astype(np.float32))
    if not images:
        raise FileNotFoundError(f"No image/mask pairs found for {dataset_key} under {data_path}")
    return np.asarray(images, dtype=np.float32), np.asarray(masks, dtype=np.float32)


def validation_subset(images, masks):
    try:
        from sklearn.model_selection import train_test_split
        _, val_images, _, val_masks = train_test_split(
            images, masks, test_size=0.1, random_state=42, shuffle=True
        )
        return val_images, val_masks
    except ModuleNotFoundError:
        rng = np.random.RandomState(42)
        indices = np.arange(len(images))
        rng.shuffle(indices)
        count = int(np.ceil(len(images) * 0.1))
        selected = indices[-count:]
        return images[selected], masks[selected]


def make_loader(data_path, dataset_key, batch_size=8, num_workers=0, input_size=352):
    images, masks = read_dataset(data_path, dataset_key, input_size)
    if dataset_key in {"Kvasir-SEG", "CVC-ClinicDB"}:
        images, masks = validation_subset(images, masks)
    dataset = PolypEvalDataset(images, masks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available()), len(dataset)


def binary_metrics_per_image(prediction, target, smooth=1e-6):
    prediction = prediction.reshape(prediction.shape[0], -1).float()
    target = target.reshape(target.shape[0], -1).float()
    intersection = (prediction * target).sum(1)
    dice = (2 * intersection + smooth) / (prediction.sum(1) + target.sum(1) + smooth)
    union = prediction.sum(1) + target.sum(1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return float(dice.mean()), float(iou.mean())


def _main_logits(output):
    if isinstance(output, dict):
        return output.get("final_logits", next(iter(output.values())))
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def tta_probability(model, images):
    """Average original, horizontal/vertical flip, and +/-10 degree predictions."""
    transforms = (
        (lambda value: value, lambda value: value),
        (lambda value: torch.flip(value, dims=[-1]), lambda value: torch.flip(value, dims=[-1])),
        (lambda value: torch.flip(value, dims=[-2]), lambda value: torch.flip(value, dims=[-2])),
        (
            lambda value: TF.rotate(value, 10, interpolation=TF.InterpolationMode.BILINEAR),
            lambda value: TF.rotate(value, -10, interpolation=TF.InterpolationMode.BILINEAR),
        ),
        (
            lambda value: TF.rotate(value, -10, interpolation=TF.InterpolationMode.BILINEAR),
            lambda value: TF.rotate(value, 10, interpolation=TF.InterpolationMode.BILINEAR),
        ),
    )
    probabilities = []
    for augment, invert in transforms:
        probabilities.append(invert(torch.sigmoid(_main_logits(model(augment(images))))))
    return torch.stack(probabilities, dim=0).mean(dim=0)


def evaluate_loader(model, loader, device, threshold=0.45, max_batches=0, use_tta=False):
    import py_sod_metrics

    sm = py_sod_metrics.Smeasure()
    wfm = py_sod_metrics.WeightedFmeasure()
    em = py_sod_metrics.Emeasure()
    mae_metric = py_sod_metrics.MAE()
    dice_values, iou_values = [], []
    model.eval()
    with torch.inference_mode():
        for batch_index, (images, masks) in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            images, masks = images.to(device), masks.to(device)
            if use_tta:
                probability = tta_probability(model, images)
            else:
                logits = _main_logits(model(images))
                probability = torch.sigmoid(logits)
            if probability.shape[-2:] != masks.shape[-2:]:
                probability = F.interpolate(
                    probability, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
            prediction = (probability > threshold).float()
            for index in range(images.shape[0]):
                dice, iou = binary_metrics_per_image(prediction[index:index + 1], masks[index:index + 1])
                dice_values.append(dice)
                iou_values.append(iou)
                pred_u8 = np.clip(probability[index, 0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
                gt_u8 = (masks[index, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
                sm.step(pred=pred_u8, gt=gt_u8)
                wfm.step(pred=pred_u8, gt=gt_u8)
                em.step(pred=pred_u8, gt=gt_u8)
                mae_metric.step(pred=pred_u8, gt=gt_u8)
    if not dice_values:
        raise RuntimeError("Evaluation loader produced no samples")
    em_curve = em.get_results()["em"]["curve"]
    return {
        "mDice": float(np.mean(dice_values)),
        "mIoU": float(np.mean(iou_values)),
        "F_beta_w": float(wfm.get_results()["wfm"]),
        "S_alpha": float(sm.get_results()["sm"]),
        "mE_phi": float(em_curve.mean()),
        "maxE_phi": float(em_curve.max()),
        "MAE": float(mae_metric.get_results()["mae"]),
    }


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total


def measure_complexity(model, input_size=352):
    from thop import profile

    device = next(model.parameters()).device
    profile_model = copy.deepcopy(model).eval()
    dummy = torch.zeros(1, 3, input_size, input_size, device=device)
    macs, _ = profile(profile_model, inputs=(dummy,), verbose=False)
    macs = int(round(macs))
    flops = 2 * macs
    return {"macs": macs, "flops": flops, "gmacs": macs / 1e9, "gflops": flops / 1e9}
