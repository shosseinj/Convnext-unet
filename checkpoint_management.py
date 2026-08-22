"""Safe single-best checkpoint persistence, validation, and resume decisions."""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class CheckpointDecision:
    action: str
    reason: str
    checkpoint_path: Path = None


def _load_valid(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        return None
    return checkpoint


def _score(checkpoint):
    return max(float(checkpoint.get("best_acc", 0.0)), float(checkpoint.get("test_iou", 0.0)))


def load_checkpoint_file(path):
    checkpoint = _load_valid(Path(path))
    if checkpoint is None:
        raise RuntimeError(f"Checkpoint cannot be loaded: {path}")
    return checkpoint


def strip_thop_state(state_dict):
    return {
        key: value for key, value in state_dict.items()
        if not key.endswith((".total_ops", ".total_params"))
    }


def atomic_save_checkpoint(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    if _load_valid(temporary) is None:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Checkpoint verification failed: {temporary}")
    temporary.replace(path)
    return path


def prepare_checkpoint(seed_dir, experiment, seed, max_epochs, log_path):
    del max_epochs, log_path  # completion is explicit checkpoint metadata
    path = Path(seed_dir) / "best_checkpoint.pth"
    if not path.exists():
        return CheckpointDecision("train", "No checkpoint found")
    checkpoint = _load_valid(path)
    if checkpoint is None:
        return CheckpointDecision("error", f"Checkpoint cannot be loaded: {path}")
    if checkpoint.get("experiment_name") != experiment.name:
        return CheckpointDecision("error", "Checkpoint experiment metadata does not match")
    if checkpoint.get("seed") != seed:
        return CheckpointDecision("error", "Checkpoint seed metadata does not match")
    if checkpoint.get("architecture") != experiment.to_dict():
        return CheckpointDecision("error", "Checkpoint architecture metadata does not match")
    if checkpoint.get("training_complete") is True:
        return CheckpointDecision("skip", "Training is complete", path)
    return CheckpointDecision("resume", "Training is incomplete", path)


def mark_training_complete(checkpoint, final_epoch, training_time_seconds, completion_reason):
    completed = dict(checkpoint)
    completed.update({
        "training_complete": True,
        "final_epoch": int(final_epoch),
        "training_time_seconds": float(training_time_seconds),
        "completion_reason": completion_reason,
    })
    return completed


def atomic_save_best(checkpoint, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pth"
    temporary_path = checkpoint_dir / "best.pth.tmp"
    torch.save(checkpoint, temporary_path)
    if _load_valid(temporary_path) is None:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Checkpoint verification failed: {temporary_path}")
    temporary_path.replace(best_path)
    for path in checkpoint_dir.glob("*.pth"):
        if path != best_path:
            path.unlink()
    return best_path


def prepare_best_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None
    best_path = checkpoint_dir / "best.pth"
    best_checkpoint = _load_valid(best_path) if best_path.is_file() else None
    candidates = []
    if best_checkpoint is not None:
        candidates.append((best_path, best_checkpoint))
    for path in sorted(checkpoint_dir.glob("*.pth")):
        if path == best_path:
            continue
        checkpoint = _load_valid(path)
        if checkpoint is not None:
            candidates.append((path, checkpoint))
    if not candidates:
        return None
    _, selected = max(candidates, key=lambda item: _score(item[1]))
    return atomic_save_best(selected, checkpoint_dir)
