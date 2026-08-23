"""Small AMP helpers shared by training and checkpoint resume paths."""

from contextlib import nullcontext

import torch


def autocast_context(enabled, device_type):
    if enabled and device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def create_grad_scaler(enabled, device_type):
    return torch.amp.GradScaler("cuda", enabled=bool(enabled and device_type == "cuda"))


def add_scaler_state(checkpoint, scaler, enabled):
    if enabled:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    return checkpoint


def restore_scaler_state(checkpoint, scaler, enabled):
    if not enabled or "scaler_state_dict" not in checkpoint:
        return False
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return True


def finish_optimizer_step(optimizer, scaler, amp_enabled, gradients_are_finite):
    if not gradients_are_finite:
        optimizer.zero_grad(set_to_none=True)
        if amp_enabled:
            scaler.update()
        return False
    if amp_enabled:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return True
