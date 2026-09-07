"""Build the final provenance-checked qualitative comparison for Exp45.

External masks come only from author-released prediction archives. Baseline and
Exp45 masks are generated through the repository's validated model path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_registry import get_experiment
from checkpoint_management import load_checkpoint_file, strip_thop_state
from evaluate import build_model
from evaluation_core import _main_logits


OUT = ROOT / "qualitative_results"
THRESHOLD = 0.45
SELECTIONS = (
    ("Kvasir-SEG", "cju2rmd2rsw9g09888hh1efu0.jpg", "small polyp"),
    ("CVC-ClinicDB", "25.png", "large polyp"),
    ("Kvasir-SEG", "cju31w6goazci0799n014ly1q.jpg", "low contrast"),
    ("CVC-ClinicDB", "374.png", "blurred or ambiguous boundary"),
    ("Kvasir-SEG", "cju2yo1j1v0qz09934o0e683p.jpg", "irregular polyp"),
    ("CVC-ClinicDB", "80.png", "confusing background / difficult case"),
)
EXTERNALS = {
    "CTNet": {
        "root": ROOT / "external_models/_extracted/CTNet/result_map",
        "repo": "https://github.com/Fhujinwu/CTNet",
        "source": "result_map.zip from the official repository",
        "artifact": ROOT / "external_models/CTNet/result_map.zip",
    },
    "MEGANet": {
        "root": ROOT / "external_models/_extracted/MEGANet/MEGANet-Res2Net",
        "repo": "https://github.com/UARK-AICV/MEGANet",
        "source": "official MEGANet-Res2Net precomputed prediction maps",
        "artifact": ROOT / "external_models/_official_artifacts/MEGANet_Res2Net_predictions.zip",
    },
    "EnFormer": {
        "root": ROOT / "external_models/_extracted/EnFormer/result_map/res/enformer",
        "repo": "https://github.com/HuangDLab/EnFormer",
        "source": "official pre-computed maps, enformer/ variant",
        "artifact": ROOT / "external_models/_official_artifacts/EnFormer_prediction_maps.zip",
    },
}
INTERNALS = {
    "Baseline": (
        "one_seed_01_baseline",
        ROOT / "one_seed_results/ablation/01_baseline/seed_42/best_checkpoint.pth",
    ),
    "Ours_Exp45": (
        "one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine",
        ROOT / "one_seed_results/ablation/45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine/seed_42/best_checkpoint.pth",
    ),
}
COLUMN_KEYS = ("Input", "GT", "Baseline", "CTNet", "MEGANet", "EnFormer", "Ours_Exp45")
COLUMN_LABELS = ("Input", "GT", "Baseline", "CTNet", "MEGANet", "EnFormer", "Ours (Exp45)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_by_stem(folder: Path, stem: str) -> Path:
    matches = sorted(p for p in folder.glob(f"{stem}.*") if p.is_file())
    if len(matches) != 1:
        raise RuntimeError(f"Expected one map for {stem} in {folder}; found {len(matches)}")
    return matches[0]


def read(path: Path, grayscale: bool = False) -> np.ndarray:
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    value = cv2.imread(str(path), mode)
    if value is None:
        raise RuntimeError(f"Cannot decode {path}")
    return value if grayscale else cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def binary_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    value = read(path, grayscale=True)
    if value.shape != shape:
        value = cv2.resize(value, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = value >= 128
    if not mask.any() or mask.all():
        raise RuntimeError(f"Degenerate all-black/all-white prediction: {path}")
    return mask.astype(np.uint8) * 255


def load_internal(label: str, device: torch.device):
    experiment, checkpoint_path = INTERNALS[label]
    config = get_experiment(experiment)
    checkpoint = load_checkpoint_file(checkpoint_path)
    if checkpoint.get("training_complete") is not True:
        raise RuntimeError(f"Incomplete checkpoint: {checkpoint_path}")
    if checkpoint.get("experiment_name") != experiment or checkpoint.get("seed") != 42:
        raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint_path}")
    if checkpoint.get("architecture") != config.to_dict():
        raise RuntimeError(f"Checkpoint architecture mismatch: {checkpoint_path}")
    model = build_model(config, ROOT / "convnext_tiny_22k_1k_384.pth", device)
    model.load_state_dict(strip_thop_state(checkpoint["model_state_dict"]), strict=True)
    return model.eval(), checkpoint, checkpoint_path


def predict(model, image_path: Path, device: torch.device) -> np.ndarray:
    image = read(image_path)
    height, width = image.shape[:2]
    resized = cv2.resize(image, (352, 352), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(_main_logits(model(tensor)))
        probability = F.interpolate(probability, (height, width), mode="bilinear", align_corners=False)
    mask = probability[0, 0].cpu().numpy() > THRESHOLD
    if not mask.any() or mask.all():
        raise RuntimeError(f"Degenerate internal prediction: {image_path}")
    return mask.astype(np.uint8) * 255


def render(rows: list[dict[str, str]], error: bool = False) -> Path:
    dpi = 400
    fig, axes = plt.subplots(len(rows), len(COLUMN_KEYS), figsize=(12.0, 9.3), facecolor="white")
    for r, row in enumerate(rows):
        rgb = read(Path(row["image_path"]))
        gt = read(Path(row["gt_path"]), grayscale=True) >= 128
        for c, key in enumerate(COLUMN_KEYS):
            ax = axes[r, c]
            ax.set_axis_off()
            if key == "Input":
                ax.imshow(rgb, interpolation="lanczos")
            elif key == "GT":
                ax.imshow(gt, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                pred = read(OUT / "predictions" / key / f"{Path(row['image_name']).stem}.png", grayscale=True) >= 128
                if error:
                    view = np.zeros((*gt.shape, 3), dtype=np.uint8)
                    view[gt & pred] = (255, 255, 255)  # true positive
                    view[~gt & pred] = (255, 0, 0)     # false positive
                    view[gt & ~pred] = (0, 170, 255)   # false negative
                    ax.imshow(view, interpolation="nearest")
                else:
                    ax.imshow(pred, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            if r == 0:
                ax.set_title(COLUMN_LABELS[c], fontsize=11, fontweight="semibold", pad=5)
        axes[r, 0].text(-0.035, 0.5, f"{row['dataset']}\n{row['case_type']}", rotation=90,
                        transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=7.2)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.965, bottom=0.008, wspace=0.025, hspace=0.035)
    if error:
        fig.text(0.5, 0.002, "Error colors: white = TP, red = FP, cyan = FN",
                 ha="center", va="bottom", fontsize=8.5)
        fig.subplots_adjust(bottom=0.025)
    name = "qualitative_results_exp45_errors.png" if error else "qualitative_results_exp45.png"
    path = OUT / name
    fig.savefig(path, dpi=dpi, facecolor="white")
    if not error:
        fig.savefig(OUT / "qualitative_results_exp45.pdf", dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ("images", "gt", *(f"predictions/{name}" for name in (*EXTERNALS, *INTERNALS))):
        (OUT / folder).mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset, image_name, case_type in SELECTIONS:
        image_path = find_by_stem(ROOT / "data" / dataset / "images", Path(image_name).stem)
        gt_path = find_by_stem(ROOT / "data" / dataset / "masks", Path(image_name).stem)
        image_out = OUT / "images" / image_path.name
        gt_out = OUT / "gt" / f"{image_path.stem}.png"
        shutil.copy2(image_path, image_out)
        shape = read(image_path).shape[:2]
        cv2.imwrite(str(gt_out), binary_mask(gt_path, shape))
        for model, spec in EXTERNALS.items():
            dataset_dir = "Kvasir" if dataset == "Kvasir-SEG" else dataset
            source = find_by_stem(spec["root"] / dataset_dir, image_path.stem)
            cv2.imwrite(str(OUT / "predictions" / model / f"{image_path.stem}.png"), binary_mask(source, shape))
        rows.append({"dataset": dataset, "image_name": image_path.name,
                     "image_path": str(image_out.resolve()), "gt_path": str(gt_out.resolve()),
                     "case_type": case_type})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    internal_meta = {}
    for label in INTERNALS:
        model, checkpoint, checkpoint_path = load_internal(label, device)
        for row in rows:
            mask = predict(model, Path(row["image_path"]), device)
            cv2.imwrite(str(OUT / "predictions" / label / f"{Path(row['image_name']).stem}.png"), mask)
        internal_meta[label] = {
            "experiment": INTERNALS[label][0], "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(checkpoint_path), "epoch": checkpoint.get("epoch"),
            "best_epoch": checkpoint.get("best_epoch"),
            "best_validation_metric": checkpoint.get("best_validation_metric"),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (OUT / "selection_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "image_name", "image_path", "gt_path", "case_type"))
        writer.writeheader()
        writer.writerows(rows)

    main_png = render(rows)
    error_png = render(rows, error=True)
    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "threshold": THRESHOLD,
        "input_size": [352, 352], "tta": False, "selected_rows": rows,
        "external_models": {name: {**{k: str(v) for k, v in spec.items() if k != "root"},
                                    "artifact_sha256": sha256(spec["artifact"])} for name, spec in EXTERNALS.items()},
        "internal_models": internal_meta,
        "column_order": list(COLUMN_LABELS),
    }
    (OUT / "qualitative_results_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(main_png)
    print(error_png)


if __name__ == "__main__":
    main()
