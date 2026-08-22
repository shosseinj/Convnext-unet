"""Structured training history and summary persistence."""

import csv
from pathlib import Path

from ablation_artifacts import atomic_write_json


HISTORY_FIELDS = (
    "epoch", "train_loss", "validation_dice", "validation_iou",
    "encoder_lr", "decoder_lr", "elapsed_seconds", "is_best",
)


def append_history_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    normalized = {field: row[field] for field in HISTORY_FIELDS}
    rows = [existing for existing in rows if int(existing["epoch"]) != int(row["epoch"])]
    rows.append(normalized)
    rows.sort(key=lambda item: int(item["epoch"]))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_training_summary(path, payload):
    atomic_write_json(path, payload)
