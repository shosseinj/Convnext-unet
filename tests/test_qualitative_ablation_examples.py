import csv
from pathlib import Path

import numpy as np
from PIL import Image

from tools.generate_qualitative_ablation_examples import DATASETS, generate


def test_full_export_has_expected_counts_and_native_sizes(tmp_path):
    data_root = tmp_path / "data"
    roots = [tmp_path / name for name in ("baseline", "second", "third")]
    expected_size = (41, 29)
    for dataset in DATASETS:
        disk_dataset = "ETIS-LARIBPOLYPDB" if dataset == "ETIS-LaribPolypDB" else dataset
        for index in range(3):
            image_id = f"case_{index}"
            image = np.full((expected_size[1], expected_size[0], 3), 40 + index * 30, dtype=np.uint8)
            mask = np.zeros((expected_size[1], expected_size[0]), dtype=np.uint8)
            mask[: 4 + index * 7, : 5 + index * 8] = 255
            image_dir = data_root / disk_dataset / "images"
            mask_dir = data_root / disk_dataset / "masks"
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(image_dir / f"{image_id}.jpg")
            Image.fromarray(mask).save(mask_dir / f"{image_id}.png")
            for root in roots:
                prediction_dir = root / disk_dataset
                prediction_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask).save(prediction_dir / f"{image_id}.png")

    output = tmp_path / "qualitative_ablation_outputs"
    summary = generate(data_root, roots[0], roots[1], roots[2], output)

    assert summary == {"datasets": 5, "samples": 10, "individual_images": 40,
                       "comparison_panels": 10, "gt_panels": 10, "skipped_files": 0}
    with (output / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    for row in rows:
        sample_dir = Path(row["original_path"]).parent
        assert {path.name for path in sample_dir.iterdir()} == {
            "original.png", "ground_truth_mask.png", "baseline_mask.png",
            "second_ablation_mask.png", "third_ablation_mask.png",
            "comparison_panel.png", "comparison_panel_with_gt.png",
        }
        for name in ("original.png", "baseline_mask.png", "second_ablation_mask.png", "third_ablation_mask.png"):
            with Image.open(sample_dir / name) as image:
                assert image.size == expected_size
