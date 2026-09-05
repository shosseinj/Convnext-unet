"""Build a ten-sample qualitative plate with equally sized panels."""
import csv
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from ablation_registry import get_experiment
from checkpoint_management import load_checkpoint_file, strip_thop_state
from evaluate import build_model
from one_seed_models import main_logits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "qualitative_ablation_outputs"
EXP = "one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine"
CKPT = ROOT / "one_seed_results/ablation/45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine/seed_42/best_checkpoint.pth"

def font(size):
    p = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(p), size) if p.exists() else ImageFont.load_default()

def fit(image, width=700, height=500, mask=False):
    image = image.convert("RGB")
    result = image.resize((width, height), Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS)
    image.close()
    return result

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_experiment(EXP)
    checkpoint = load_checkpoint_file(CKPT)
    if checkpoint.get("experiment_name") != EXP or checkpoint.get("seed") != 42 or checkpoint.get("architecture") != config.to_dict():
        raise RuntimeError("Experiment/checkpoint provenance mismatch")
    model = build_model(config, ROOT / "convnext_tiny_22k_1k_384.pth", device)
    model.load_state_dict(strip_thop_state(checkpoint["model_state_dict"]), strict=True)
    model.eval()
    values = {}
    def hook(_module, _inputs, output):
        values["activation"] = output
        output.register_hook(lambda grad: values.__setitem__("gradient", grad))
    handle = model.residual_fg_mscb_lite_stage3.register_forward_hook(hook)
    with (OUT / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10:
        raise RuntimeError(f"Expected 10 rows, found {len(rows)}")
    overlays = []
    for row in rows:
        bgr = cv2.imread(str(Path(row["original_path"])), cv2.IMREAD_COLOR)
        original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(cv2.resize(original, (352, 352)).transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        logits = main_logits(model(tensor))
        fg = (torch.sigmoid(logits.detach()) >= 0.45).float()
        if not fg.any():
            fg = (logits.detach() >= torch.quantile(logits.detach().flatten(), 0.9)).float()
        ((logits * fg).sum() / fg.sum().clamp_min(1)).backward()
        act, grad = values["activation"].detach(), values["gradient"].detach()
        cam = torch.relu((grad.mean((2, 3), keepdim=True) * act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=original.shape[:2], mode="bilinear", align_corners=False)[0, 0]
        cam = ((cam - cam.min()) / (cam.max() - cam.min()).clamp_min(1e-8)).cpu().numpy()
        color = cv2.cvtColor(cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        alpha = np.clip(cam[..., None] * 0.65, 0.12, 0.65)
        overlays.append(Image.fromarray(np.uint8(original * (1 - alpha) + color * alpha)))
    handle.remove()
    titles = ("Original", "Ground Truth", "Baseline", "FAFEM + MSCB", "Residual RFG-MSCB", "Activation Heatmap")
    sources = ("original_path", "ground_truth_mask.png", "baseline_mask_path", "second_ablation_mask_path", "third_ablation_mask_path")
    label_w, cell_w, cell_h, outer, gap, header, row_gap = 105, 700, 500, 25, 20, 90, 22
    width = outer * 2 + label_w + 6 * cell_w + 5 * gap
    height = outer * 2 + header + 10 * cell_h + 9 * row_gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, label_font = font(40), font(31)
    for col, title in enumerate(titles):
        x = outer + label_w + col * (cell_w + gap)
        box = draw.textbbox((0, 0), title, font=title_font)
        draw.text((x + (cell_w - box[2] + box[0]) / 2, outer + 13), title, fill=(20, 20, 20), font=title_font)
    draw.line((outer, outer + header, width - outer, outer + header), fill=(50, 50, 50), width=4)
    previous = None
    for index, (row, overlay) in enumerate(zip(rows, overlays)):
        top = outer + header + index * (cell_h + row_gap)
        if previous and row["dataset"] != previous:
            draw.line((outer, top - row_gap // 2, width - outer, top - row_gap // 2), fill=(45, 45, 45), width=5)
        previous = row["dataset"]
        label = f"{row['dataset']}  |  {int(row['sample_index']):02d}"
        label_img = Image.new("RGB", (cell_h, label_w), "white")
        ld = ImageDraw.Draw(label_img); box = ld.textbbox((0, 0), label, font=label_font)
        ld.text(((cell_h - box[2] + box[0]) / 2, 31), label, fill=(20, 20, 20), font=label_font)
        rotated = label_img.rotate(90, expand=True); canvas.paste(rotated, (outer, top)); label_img.close(); rotated.close()
        sample_dir = Path(row["original_path"]).parent
        for col, name in enumerate(sources):
            path = sample_dir / name if name.endswith(".png") and name not in row else Path(row[name])
            with Image.open(path) as image:
                cell = fit(image, cell_w, cell_h, col > 0)
            canvas.paste(cell, (outer + label_w + col * (cell_w + gap), top)); cell.close()
        cell = fit(overlay, cell_w, cell_h); canvas.paste(cell, (outer + label_w + 5 * (cell_w + gap), top)); cell.close(); overlay.close()
        if index < 9:
            y = top + cell_h + row_gap // 2; draw.line((outer, y, width - outer, y), fill=(95, 95, 95), width=2)
    output = OUT / "ablation_img_10_with_heatmap.png"
    canvas.save(output, "PNG", compress_level=1, dpi=(300, 300)); canvas.close()
    print(f"Saved {output} ({width}x{height}); every panel={cell_w}x{cell_h}; samples=10; device={device}")

if __name__ == "__main__":
    main()
