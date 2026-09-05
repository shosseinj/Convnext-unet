#!/usr/bin/env python3
"""Create deterministic, publication-ready qualitative ablation comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


DATASETS = (
    "Kvasir-SEG",
    "CVC-ClinicDB",
    "CVC-300",
    "CVC-ColonDB",
    "ETIS-LaribPolypDB",
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MANIFEST_COLUMNS = (
    "dataset",
    "sample_index",
    "image_id",
    "original_path",
    "baseline_mask_path",
    "second_ablation_mask_path",
    "third_ablation_mask_path",
    "comparison_panel_path",
    "comparison_panel_with_gt_path",
)


@dataclass(frozen=True)
class Candidate:
    image_id: str
    original: Path
    baseline: Path
    second: Path
    third: Path
    ground_truth: Path | None
    area_ratio: float | None
    size: tuple[int, int]


def _dataset_dir(root: Path, dataset: str) -> Path:
    """Resolve a dataset directory while tolerating the repository's ETIS casing."""
    direct = root / dataset
    if direct.is_dir():
        return direct
    matches = [entry for entry in root.iterdir() if entry.is_dir() and entry.name.casefold() == dataset.casefold()]
    if len(matches) == 1:
        return matches[0]
    return direct


def _index_images(directory: Path, label: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"Missing {label} directory: {directory}")
    indexed: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem.casefold()
        if key in indexed:
            raise ValueError(f"Duplicate image ID {path.stem!r} in {directory}")
        indexed[key] = path
    if not indexed:
        raise ValueError(f"No supported images found in {label} directory: {directory}")
    return indexed


def _open_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size
    except UnidentifiedImageError:
        decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise
        return decoded.shape[1], decoded.shape[0]


def _mask_area(path: Path) -> float:
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"))
    return float(np.count_nonzero(values >= 128) / values.size)


def _valid_candidates(
    dataset: str,
    data_root: Path,
    prediction_roots: Sequence[Path],
    ground_truth_root: Path | None,
) -> tuple[list[Candidate], list[str]]:
    image_dir = _dataset_dir(data_root, dataset) / "images"
    originals = _index_images(image_dir, f"{dataset} originals")
    predictions = [
        _index_images(_dataset_dir(root, dataset), f"{dataset} {name} predictions")
        for root, name in zip(prediction_roots, ("baseline", "second ablation", "third ablation"))
    ]
    gt_base = _dataset_dir(ground_truth_root or data_root, dataset)
    gt_dir = gt_base / "masks" if (gt_base / "masks").is_dir() else gt_base
    ground_truth = _index_images(gt_dir, f"{dataset} ground truth") if gt_dir.is_dir() else {}

    common = set(originals)
    for prediction in predictions:
        common &= set(prediction)
    skipped = sorted(set(originals) - common)
    candidates: list[Candidate] = []
    for key in sorted(common):
        paths = [originals[key], predictions[0][key], predictions[1][key], predictions[2][key]]
        gt_path = ground_truth.get(key)
        try:
            sizes = [_open_size(path) for path in paths]
            if len(set(sizes)) != 1:
                skipped.append(f"{originals[key].name} (spatial-size mismatch: {sizes})")
                continue
            area = None
            if gt_path is not None:
                if _open_size(gt_path) != sizes[0]:
                    gt_path = None
                else:
                    area = _mask_area(gt_path)
            candidates.append(Candidate(
                image_id=originals[key].stem,
                original=originals[key], baseline=predictions[0][key],
                second=predictions[1][key], third=predictions[2][key],
                ground_truth=gt_path, area_ratio=area, size=sizes[0],
            ))
        except (OSError, UnidentifiedImageError, ValueError) as error:
            skipped.append(f"{originals[key].name} ({error})")
    return candidates, skipped


def deterministic_selection(rows: Sequence[Mapping[str, object]], count: int = 1) -> list[Mapping[str, object]]:
    """Select rows nearest evenly spaced GT-area quantiles, breaking ties by ID."""
    if count < 1 or len(rows) < count:
        raise ValueError(f"Need at least {count} candidates; found {len(rows)}")
    ranked = sorted(rows, key=lambda row: (float(row.get("mask_area_ratio", 0.0)), str(row.get("image", "")).casefold()))
    remaining = list(ranked)
    selected: list[Mapping[str, object]] = []
    for quantile in ((index + 1) / (count + 1) for index in range(count)):
        target = quantile * (len(ranked) - 1)
        chosen = min(remaining, key=lambda row: (abs(ranked.index(row) - target), str(row.get("image", "")).casefold()))
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _select_candidates(candidates: Sequence[Candidate], count: int) -> list[Candidate]:
    with_gt = [candidate for candidate in candidates if candidate.area_ratio is not None]
    pool = with_gt if len(with_gt) >= count else list(candidates)
    rows = [{"image": item.image_id, "mask_area_ratio": item.area_ratio or 0.0, "candidate": item} for item in pool]
    return [row["candidate"] for row in deterministic_selection(rows, count)]  # type: ignore[list-item]


def _save_original(source: Path, destination: Path) -> None:
    try:
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            rgb.save(destination, format="PNG", compress_level=1, dpi=(300, 300))
    except UnidentifiedImageError:
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if bgr is None:
            raise
        Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(
            destination, format="PNG", compress_level=1, dpi=(300, 300)
        )


def _load_binary_mask(source: Path) -> Image.Image:
    with Image.open(source) as image:
        values = np.asarray(image.convert("L"))
    return Image.fromarray(np.where(values >= 128, 255, 0).astype(np.uint8), mode="L")


def _save_mask(source: Path, destination: Path) -> None:
    _load_binary_mask(source).save(destination, format="PNG", compress_level=1, dpi=(300, 300))


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _panel(images: Sequence[Path], titles: Sequence[str], destination: Path) -> None:
    opened: list[Image.Image] = []
    try:
        for path in images:
            with Image.open(path) as source:
                opened.append(source.convert("RGB"))
        cell_width = max(image.width for image in opened)
        cell_height = max(image.height for image in opened)
        margin, gap, title_height = 24, 18, 54
        canvas = Image.new("RGB", (2 * margin + len(opened) * cell_width + (len(opened) - 1) * gap,
                                   2 * margin + title_height + cell_height), "white")
        draw = ImageDraw.Draw(canvas)
        font = _font(max(18, min(30, cell_width // 16)))
        for index, (image, title) in enumerate(zip(opened, titles)):
            x = margin + index * (cell_width + gap)
            y = margin + title_height
            canvas.paste(image, (x + (cell_width - image.width) // 2, y + (cell_height - image.height) // 2))
            box = draw.textbbox((0, 0), title, font=font)
            draw.text((x + (cell_width - (box[2] - box[0])) / 2, margin), title, fill="black", font=font)
        canvas.save(destination, format="PNG", compress_level=1, dpi=(300, 300))
    finally:
        for image in opened:
            image.close()


def generate(
    data_root: Path,
    baseline_root: Path,
    second_root: Path,
    third_root: Path,
    output_root: Path,
    ground_truth_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    preflight: dict[str, tuple[list[Candidate], list[str]]] = {}
    for dataset in DATASETS:
        candidates, skipped = _valid_candidates(
            dataset, data_root, (baseline_root, second_root, third_root), ground_truth_root
        )
        if len(candidates) < 2:
            raise ValueError(f"{dataset}: need 2 valid common samples; found {len(candidates)}")
        preflight[dataset] = (_select_candidates(candidates, 2), skipped)

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_root} (use --overwrite)")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    panel_count = 0
    gt_panel_count = 0
    skipped_count = sum(len(value[1]) for value in preflight.values())
    for dataset, (selected, _skipped) in preflight.items():
        for sample_index, candidate in enumerate(selected, start=1):
            sample_dir = output_root / dataset / f"sample_{sample_index:02d}_{candidate.image_id}"
            sample_dir.mkdir(parents=True)
            output_paths = {
                "original_path": sample_dir / "original.png",
                "baseline_mask_path": sample_dir / "baseline_mask.png",
                "second_ablation_mask_path": sample_dir / "second_ablation_mask.png",
                "third_ablation_mask_path": sample_dir / "third_ablation_mask.png",
            }
            _save_original(candidate.original, output_paths["original_path"])
            _save_mask(candidate.baseline, output_paths["baseline_mask_path"])
            _save_mask(candidate.second, output_paths["second_ablation_mask_path"])
            _save_mask(candidate.third, output_paths["third_ablation_mask_path"])
            panel_path = sample_dir / "comparison_panel.png"
            _panel(list(output_paths.values()), ("Original", "Baseline", "Second Ablation", "Third Ablation"), panel_path)
            panel_count += 1
            gt_panel_path: Path | None = None
            if candidate.ground_truth is not None:
                gt_path = sample_dir / "ground_truth_mask.png"
                _save_mask(candidate.ground_truth, gt_path)
                gt_panel_path = sample_dir / "comparison_panel_with_gt.png"
                _panel((output_paths["original_path"], gt_path, output_paths["baseline_mask_path"],
                        output_paths["second_ablation_mask_path"], output_paths["third_ablation_mask_path"]),
                       ("Original", "GT", "Baseline", "Second Ablation", "Third Ablation"), gt_panel_path)
                gt_panel_count += 1
            rows.append({
                "dataset": dataset, "sample_index": sample_index, "image_id": candidate.image_id,
                **{key: str(value.resolve()) for key, value in output_paths.items()},
                "comparison_panel_path": str(panel_path.resolve()),
                "comparison_panel_with_gt_path": str(gt_panel_path.resolve()) if gt_panel_path else "",
            })
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"datasets": len(DATASETS), "samples": len(rows), "individual_images": len(rows) * 4,
               "comparison_panels": panel_count, "gt_panels": gt_panel_count, "skipped_files": skipped_count}
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional JSON file containing the path arguments")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--second-ablation-root", type=Path)
    parser.add_argument("--third-ablation-root", type=Path)
    parser.add_argument("--ground-truth-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config: dict[str, object] = {}
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    def path_arg(name: str, required: bool = True, default: Path | None = None) -> Path | None:
        value = getattr(args, name) or config.get(name)
        if value is None:
            value = default
        if value is None and required:
            raise ValueError(f"Missing --{name.replace('_', '-')}; provide it directly or in --config")
        return Path(value).expanduser().resolve() if value is not None else None
    try:
        summary = generate(
            data_root=path_arg("data_root"),  # type: ignore[arg-type]
            baseline_root=path_arg("baseline_root"),  # type: ignore[arg-type]
            second_root=path_arg("second_ablation_root"),  # type: ignore[arg-type]
            third_root=path_arg("third_ablation_root"),  # type: ignore[arg-type]
            ground_truth_root=path_arg("ground_truth_root", required=False),
            output_root=path_arg("output_root", default=Path("qualitative_ablation_outputs")),  # type: ignore[arg-type]
            overwrite=args.overwrite or bool(config.get("overwrite", False)),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Datasets processed: {summary['datasets']}")
    print(f"Selected samples: {summary['samples']}")
    print(f"Individual images saved: {summary['individual_images']}")
    print(f"Comparison panels saved: {summary['comparison_panels']} (+ {summary['gt_panels']} with GT)")
    print(f"Skipped files/candidates: {summary['skipped_files']}")
    print("Count note: 2 samples x 5 datasets x 4 image types = 40 individual images (not 30).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
