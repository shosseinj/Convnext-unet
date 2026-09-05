"""Generate real Grad-CAM activation overlays for Experiment 45, seed 42."""

from pathlib import Path
import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ablation_registry import get_experiment
from checkpoint_management import load_checkpoint_file, strip_thop_state
from evaluate import build_model
from one_seed_models import main_logits


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine"
CHECKPOINT = ROOT / "one_seed_results/ablation/45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine/seed_42/best_checkpoint.pth"
MANIFEST = ROOT / "qualitative_results/selection_manifest.csv"
OUT = ROOT / "qualitative_results"


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot decode {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_experiment(EXPERIMENT)
    checkpoint = load_checkpoint_file(CHECKPOINT)
    if checkpoint.get("experiment_name") != EXPERIMENT or checkpoint.get("seed") != 42:
        raise RuntimeError("Experiment/checkpoint provenance mismatch")
    if checkpoint.get("architecture") != config.to_dict():
        raise RuntimeError("Checkpoint architecture does not match registered Experiment 45")
    model = build_model(config, ROOT / "convnext_tiny_22k_1k_384.pth", device)
    model.load_state_dict(strip_thop_state(checkpoint["model_state_dict"]), strict=True)
    model.eval()
    layer = model.residual_fg_mscb_lite_stage3
    if layer is None:
        raise RuntimeError("Experiment 45 residual_fg_mscb_lite_stage3 is unavailable")

    activation = {}
    gradient = {}
    def forward_hook(_module, _inputs, output):
        activation["value"] = output
        output.register_hook(lambda grad: gradient.__setitem__("value", grad))
    handle = layer.register_forward_hook(forward_hook)

    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    results = []
    for row in rows:
        image_path = ROOT / "data" / row["dataset"] / "images" / row["filename"]
        gt_path = ROOT / "data" / row["dataset"] / "masks" / row["filename"]
        original = read_rgb(image_path)
        resized = cv2.resize(original, (352, 352), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        logits = main_logits(model(tensor))
        foreground = (torch.sigmoid(logits.detach()) >= 0.45).float()
        if not foreground.any():
            cutoff = torch.quantile(logits.detach().flatten(), 0.9)
            foreground = (logits.detach() >= cutoff).float()
        target = (logits * foreground).sum() / foreground.sum().clamp_min(1)
        target.backward()
        acts, grads = activation["value"].detach(), gradient["value"].detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=original.shape[:2], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        prediction = F.interpolate(torch.sigmoid(logits), size=original.shape[:2], mode="bilinear", align_corners=False)[0, 0]
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        results.append((original, gt, cam.cpu().numpy(), (prediction >= 0.45).cpu().numpy()))
        np.save(OUT / "Ours" / f"{Path(row['filename']).stem}_gradcam.npy", cam.cpu().numpy())
    handle.remove()

    fig, axes = plt.subplots(len(results), 4, figsize=(7.2, 10.0), facecolor="white")
    titles = ["Image", "GT", "Ours Activation", "Ours Prediction"]
    for r, (image, gt, cam, pred) in enumerate(results):
        axes[r, 0].imshow(image)
        axes[r, 1].imshow(gt, cmap="gray", vmin=0, vmax=255)
        axes[r, 2].imshow(image)
        axes[r, 2].imshow(cam, cmap="jet", vmin=0, vmax=1, alpha=np.clip(cam * 0.65, 0.12, 0.65))
        axes[r, 3].imshow(pred, cmap="gray", vmin=0, vmax=1)
        for c in range(4):
            axes[r, c].set_axis_off()
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10, pad=5)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.965, bottom=0.01, wspace=0.025, hspace=0.035)
    for name in ("qualitative_comparison", "exp45_activation_heatmaps"):
        fig.savefig(OUT / f"{name}.png", dpi=400, facecolor="white")
        fig.savefig(OUT / f"{name}.pdf", dpi=400, facecolor="white")
    plt.close(fig)

    report = f"""# Experiment 45 activation heatmaps

- Experiment: `{EXPERIMENT}`
- Seed: `42`
- Checkpoint: `{CHECKPOINT}`
- Checkpoint architecture validation: PASS
- Activation method: Grad-CAM using foreground logits at threshold 0.45
- Hooked module: `ConvNeXtUNet.residual_fg_mscb_lite_stage3`
- Raw activation arrays: `qualitative_results/Ours/*_gradcam.npy`
- Figure: `qualitative_results/qualitative_comparison.png`
- PDF: `qualitative_results/qualitative_comparison.pdf`
"""
    (ROOT / "heatmap_exp45_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
