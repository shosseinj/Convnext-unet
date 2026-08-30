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
    extra_fields = tuple(field for field in row if field not in HISTORY_FIELDS)
    existing_extra_fields = tuple(
        field for field in (rows[0].keys() if rows else ())
        if field not in HISTORY_FIELDS
    )
    fieldnames = HISTORY_FIELDS + tuple(
        dict.fromkeys(existing_extra_fields + extra_fields)
    )
    normalized = {field: row.get(field, "") for field in fieldnames}
    rows = [existing for existing in rows if int(existing["epoch"]) != int(row["epoch"])]
    rows.append(normalized)
    rows.sort(key=lambda item: int(item["epoch"]))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_training_summary(path, payload):
    atomic_write_json(path, payload)
