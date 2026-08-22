"""Safe single-best checkpoint persistence and legacy migration."""

from pathlib import Path

import torch


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
