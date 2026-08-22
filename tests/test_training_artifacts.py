import csv
import json
import tempfile
import unittest
from pathlib import Path

from training_artifacts import append_history_row, write_training_summary


class TrainingArtifactTests(unittest.TestCase):
    def test_history_deduplicates_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_history.csv"
            row = {
                "epoch": 4, "train_loss": 0.2, "validation_dice": 0.8,
                "validation_iou": 0.7, "encoder_lr": 1e-5,
                "decoder_lr": 1e-4, "elapsed_seconds": 12.0, "is_best": True,
            }
            append_history_row(path, row)
            append_history_row(path, row)
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["epoch"], "4")

    def test_training_summary_is_atomic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_summary.json"
            write_training_summary(path, {"experiment_name": "01_baseline", "seed": 42})
            self.assertEqual(json.loads(path.read_text())["seed"], 42)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
