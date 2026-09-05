"""Build the provenance-preserving external-model qualitative comparison figure."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "qualitative_results"
MODELS = ["CTNet", "MEGANet", "PraNetV2", "MLBNet", "CIFFormer"]
COLUMNS = ["Input", "GT", "CTNet", "MEGANet", "PraNet-V2", "MLB-Net", "CIFFormer"]

SOURCES = {
    "CTNet": {
        "Kvasir-SEG": ROOT / "external_models/_extracted/CTNet/result_map/Kvasir",
        "CVC-ClinicDB": ROOT / "external_models/_extracted/CTNet/result_map/CVC-ClinicDB",
    },
    "MEGANet": {
        "Kvasir-SEG": ROOT / "external_models/_extracted/MEGANet/MEGANet-Res2Net/Kvasir",
    },
}


def find_by_stem(folder: Path, stem: str) -> Path | None:
    matches = sorted(p for p in folder.glob(f"{stem}.*") if p.is_file())
    return matches[0] if matches else None


def features(image_path: Path, mask_path: Path) -> dict[str, float]:
    rgb = load_image(image_path, mask=False).astype(np.float32) / 255.0
    gray = rgb.mean(axis=2)
    mask = load_image(mask_path, mask=True) > 127
    area = float(mask.mean())
    inside = float(gray[mask].mean()) if mask.any() else 0.0
    outside = float(gray[~mask].mean()) if (~mask).any() else 0.0
    contrast = abs(inside - outside)
    blurred = np.asarray(Image.fromarray((gray * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    sharpness = float(blurred.var())
    boundary = mask ^ np.pad(mask[1:-1, 1:-1], 1)
    perimeter = float(boundary.sum())
    irregularity = perimeter * perimeter / max(float(mask.sum()), 1.0)
    background = float(gray[~mask].std()) if (~mask).any() else 0.0
    return {"area": area, "contrast": contrast, "sharpness": sharpness,
            "irregularity": irregularity, "background": background}


def load_image(path: Path, mask: bool) -> np.ndarray:
    """Read normal images and ClinicDB's TIFF payloads with .png names."""
    mode = cv2.IMREAD_GRAYSCALE if mask else cv2.IMREAD_COLOR
    array = cv2.imread(str(path), mode)
    if array is None:
        raise ValueError(f"OpenCV could not decode {path}")
    return array if mask else cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def candidates(dataset: str) -> list[dict]:
    image_dir = ROOT / "data" / dataset / "images"
    mask_dir = ROOT / "data" / dataset / "masks"
    ct_dir = SOURCES["CTNet"][dataset]
    rows = []
    for ct in sorted(ct_dir.iterdir()):
        image = find_by_stem(image_dir, ct.stem)
        mask = find_by_stem(mask_dir, ct.stem)
        if image and mask:
            rows.append({"dataset": dataset, "image": image, "mask": mask,
                         "stem": image.stem, **features(image, mask)})
    return rows


def choose(rows: list[dict], key: str, reverse: bool, used: set[tuple[str, str]]) -> dict:
    for row in sorted(rows, key=lambda r: r[key], reverse=reverse):
        identity = (row["dataset"], row["stem"])
        if identity not in used:
            used.add(identity)
            return row
    raise RuntimeError(f"No unused candidate for {key}")


def show_contained(ax, image: np.ndarray) -> None:
    ax.set_facecolor("white")
    ax.imshow(image, aspect="equal", interpolation="nearest")
    ax.set_axis_off()


def show_overlay(ax, image: np.ndarray, heatmap: np.ndarray) -> None:
    """Overlay a fixed-scale heatmap on its aligned original image."""
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    strength = heatmap.astype(np.float32) / 255.0
    alpha = np.where(strength > 0.02, 0.22 + 0.53 * strength, 0.0)
    ax.set_facecolor("white")
    ax.imshow(image, aspect="equal", interpolation="nearest")
    ax.imshow(heatmap, cmap="turbo", vmin=0, vmax=255, alpha=alpha,
              aspect="equal", interpolation="bilinear")
    ax.set_axis_off()


def main() -> None:
    for folder in ["images", "gt", *MODELS]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)

    kva = candidates("Kvasir-SEG")
    cvc = candidates("CVC-ClinicDB")
    used: set[tuple[str, str]] = set()
    selections = [
        ("small polyp", choose(kva, "area", False, used)),
        ("large polyp", choose(cvc, "area", True, used)),
        ("low contrast", choose(kva, "contrast", False, used)),
        ("blurred/ambiguous boundary", choose(cvc, "sharpness", False, used)),
        ("irregular shape", choose(kva, "irregularity", True, used)),
        ("difficult background", choose(cvc, "background", True, used)),
    ]

    manifest_rows = []
    panels = []
    for category, row in selections:
        filename = row["image"].name
        shutil.copy2(row["image"], OUT / "images" / filename)
        gt_out = OUT / "gt" / filename
        cv2.imwrite(str(gt_out), load_image(row["mask"], mask=True))
        model_paths: dict[str, Path | None] = {}
        for model in MODELS:
            source_dir = SOURCES.get(model, {}).get(row["dataset"])
            source = find_by_stem(source_dir, row["stem"]) if source_dir else None
            destination = OUT / model / filename
            if source:
                cv2.imwrite(str(destination), load_image(source, mask=True))
                model_paths[model] = destination
            else:
                model_paths[model] = None
        panels.append((row, gt_out, model_paths))
        manifest_rows.append({
            "category": category, "dataset": row["dataset"], "filename": filename,
            **{k: f"{row[k]:.6f}" for k in ("area", "contrast", "sharpness", "irregularity", "background")},
        })

    with (OUT / "selection_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    fig, axes = plt.subplots(len(panels), len(COLUMNS), figsize=(12.2, 10.4), facecolor="white")
    for r, (row, gt_path, model_paths) in enumerate(panels):
        entries = [row["image"], gt_path, *[model_paths[m] for m in MODELS]]
        original = load_image(row["image"], mask=False)
        for c, path in enumerate(entries):
            ax = axes[r, c]
            if path is None:
                ax.set_facecolor("white")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", color="0.45", fontsize=10)
                ax.set_axis_off()
            else:
                if c == 0:
                    show_contained(ax, original)
                else:
                    show_overlay(ax, original, load_image(path, mask=True))
            if r == 0:
                ax.set_title(COLUMNS[c], fontsize=10, pad=5)
        axes[r, 0].text(-0.03, 0.5, row["dataset"], rotation=90, va="center", ha="right",
                        transform=axes[r, 0].transAxes, fontsize=7.5, color="0.25")
    fig.subplots_adjust(left=0.065, right=0.995, top=0.965, bottom=0.01, wspace=0.025, hspace=0.035)
    fig.savefig(OUT / "qualitative_comparison.png", dpi=400, facecolor="white")
    fig.savefig(OUT / "qualitative_comparison.pdf", dpi=400, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
