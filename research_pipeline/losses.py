"""Shared segmentation losses, including the UGBR composite objective."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def boundary_weights(target, kappa):
    dilated = F.max_pool2d(target, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-target, 3, stride=1, padding=1)
    return 1.0 + kappa * (dilated - eroded).clamp(0, 1)


def morphological_gradient_target(target, kernel_size=3):
    """Return a binary dilation-minus-erosion boundary target."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3")
    padding = kernel_size // 2
    dilated = F.max_pool2d(target, kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-target, kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp(0, 1)


class DiceBCEBoundaryLoss(nn.Module):
    def __init__(self, dice_weight=0.30, bce_weight=0.30, boundary_weight=0.40,
                 boundary_kappa=5.0, label_smoothing=0.02):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.boundary_weight = boundary_weight
        self.boundary_kappa = boundary_kappa
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        probability = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probability * target).sum(dims)
        dice_loss = 1 - ((2 * intersection + 1.0) /
                         (probability.sum(dims) + target.sum(dims) + 1.0)).mean()
        smooth_target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(logits, smooth_target)
        weights = boundary_weights(target, self.boundary_kappa)
        boundary_bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        boundary_bce = (boundary_bce * weights).sum() / weights.sum().clamp_min(1e-6)
        weighted_intersection = (weights * target * probability).sum(dims)
        weighted_union = (weights * (target + probability - target * probability)).sum(dims)
        boundary_iou = (1 - (weighted_intersection + 1e-6) / (weighted_union + 1e-6)).mean()
        boundary = boundary_bce + boundary_iou
        return self.dice_weight * dice_loss + self.bce_weight * bce + self.boundary_weight * boundary


def supervised_loss(outputs, target, criterion, weights=(1.0, 0.1, 0.05, 0.02)):
    tensors = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
    return sum(weight * criterion(output, target) for output, weight in zip(tensors, weights))


def ugbr_composite_loss(outputs, target, segmentation_criterion,
                        uncertainty_threshold=0.5):
    """Compute the contracted UGBR loss and expose every finite component.

    Consistency penalizes final-vs-initial probability changes only where the
    initial prediction is confident (i.e. outside its uncertain region).
    """
    required = {"initial_logits", "boundary_logits", "final_logits", "uncertainty"}
    missing = required.difference(outputs)
    if missing:
        raise KeyError(f"UGBR output is missing required keys: {sorted(missing)}")

    final_seg = segmentation_criterion(outputs["final_logits"], target)
    initial_seg = segmentation_criterion(outputs["initial_logits"], target)
    boundary_target = morphological_gradient_target(target)
    boundary = F.binary_cross_entropy_with_logits(outputs["boundary_logits"], boundary_target)

    confident_mask = (outputs["uncertainty"].detach() < uncertainty_threshold).to(target.dtype)
    probability_change = (
        torch.sigmoid(outputs["final_logits"]) -
        torch.sigmoid(outputs["initial_logits"])
    ).square()
    consistency = (probability_change * confident_mask).sum() / confident_mask.sum().clamp_min(1.0)
    total = final_seg + 0.4 * initial_seg + 0.2 * boundary + 0.1 * consistency
    components = {
        "total": total,
        "seg_final": final_seg,
        "seg_initial": initial_seg,
        "boundary": boundary,
        "consistency": consistency,
    }
    if not all(torch.isfinite(value).all() for value in components.values()):
        raise RuntimeError("Non-finite UGBR loss component")
    return components
