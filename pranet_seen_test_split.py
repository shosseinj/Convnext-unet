"""Deterministic seen-test and internal-validation partitioning for final runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from sklearn.model_selection import train_test_split


SPLIT_PROTOCOL = "pranet_seen_test_internal_validation_v1"
SPLIT_DATASETS = ("Kvasir-SEG", "CVC-ClinicDB")
EXPECTED_COUNTS = {
    "Kvasir-SEG": {"all": 1000, "pool": 900, "final_seen_test": 100},
    "CVC-ClinicDB": {"all": 612, "pool": 550, "final_seen_test": 62},
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def discover_paired_filenames(data_root: Path | str, dataset_name: str) -> list[str]:
    """Return sorted basenames that have readable-format image and mask files."""
    dataset_root = Path(data_root) / dataset_name
    image_dir = dataset_root / "images"
    mask_dir = dataset_root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing images/masks directory for {dataset_name}")
    return sorted(
        path.name
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and (mask_dir / path.name).is_file()
    )


def _manifest_filenames(manifest: Mapping, dataset_name: str, key: str) -> list[str]:
    try:
        return [entry["image"] for entry in manifest["datasets"][dataset_name][key]]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Split manifest lacks {dataset_name}.{key}") from exc


def build_seen_test_split(
    manifest_path: Path | str,
    available_filenames: Mapping[str, Sequence[str]],
    internal_validation_seed: int = 42,
) -> dict:
    """Build the fixed seen-test and seed-independent internal split.

    The input manifest's historical ``validation`` entries are deliberately
    renamed to ``final_seen_test`` here: they are never returned as training
    or checkpoint-selection candidates.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets: dict[str, dict[str, list[str]]] = {}
    combined_pool: list[str] = []

    for dataset_name in SPLIT_DATASETS:
        expected = EXPECTED_COUNTS[dataset_name]
        available = list(available_filenames.get(dataset_name, ()))
        if len(available) != expected["all"] or len(set(available)) != len(available):
            raise ValueError(
                f"{dataset_name} must provide exactly {expected['all']} unique image/mask pairs; "
                f"found {len(available)}"
            )
        pool = _manifest_filenames(manifest, dataset_name, "train")
        final_seen_test = _manifest_filenames(manifest, dataset_name, "validation")
        if len(pool) != expected["pool"] or len(final_seen_test) != expected["final_seen_test"]:
            raise ValueError(f"Unexpected {dataset_name} manifest partition counts")
        if len(set(pool)) != len(pool) or len(set(final_seen_test)) != len(final_seen_test):
            raise ValueError(f"Duplicate filenames in {dataset_name} manifest partition")
        if set(pool) & set(final_seen_test):
            raise ValueError(f"Overlapping pool and final seen test for {dataset_name}")
        if set(pool) | set(final_seen_test) != set(available):
            raise ValueError(f"{dataset_name} manifest does not exactly cover local paired files")

        datasets[dataset_name] = {
            "pool": pool,
            "final_seen_test": final_seen_test,
        }
        combined_pool.extend(f"{dataset_name}/{name}" for name in pool)

    train, internal_validation = train_test_split(
        combined_pool,
        test_size=0.1,
        random_state=internal_validation_seed,
        shuffle=True,
    )
    if len(train) != 1305 or len(internal_validation) != 145:
        raise ValueError("Expected exactly 1305 internal-train and 145 internal-validation samples")
    if set(train) & set(internal_validation):
        raise ValueError("Internal training and validation overlap")

    final_seen_tokens = {
        f"{dataset_name}/{name}"
        for dataset_name, partitions in datasets.items()
        for name in partitions["final_seen_test"]
    }
    if set(train) & final_seen_tokens or set(internal_validation) & final_seen_tokens:
        raise ValueError("Final seen-test samples leaked into training or internal validation")

    return {
        "protocol": SPLIT_PROTOCOL,
        "source_manifest": str(manifest_path.as_posix()),
        "internal_validation_seed": internal_validation_seed,
        **datasets,
        "combined": {"train": sorted(train), "internal_validation": sorted(internal_validation)},
    }


def filenames_for_partition(split: Mapping, dataset_name: str, partition: str) -> list[str]:
    """Return local basenames for a per-dataset pool or final seen-test partition."""
    if partition not in {"pool", "final_seen_test"}:
        raise ValueError(f"Unsupported per-dataset partition: {partition}")
    return list(split[dataset_name][partition])


def combined_partition_filenames(split: Mapping, dataset_name: str, partition: str) -> list[str]:
    """Return the requested internal-train/internal-validation basenames for one dataset."""
    if partition not in {"train", "internal_validation"}:
        raise ValueError(f"Unsupported combined partition: {partition}")
    prefix = f"{dataset_name}/"
    return [token[len(prefix):] for token in split["combined"][partition] if token.startswith(prefix)]


def write_split_report(split: Mapping, output_path: Path | str) -> None:
    """Persist exact partition IDs for a run without changing the source manifest."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
