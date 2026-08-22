import tempfile
import unittest
from pathlib import Path

import torch

from checkpoint_management import atomic_save_best, prepare_best_checkpoint


class CheckpointManagementTests(unittest.TestCase):
    def checkpoint(self, epoch, score):
        return {
            "epoch": epoch,
            "best_acc": score,
            "test_iou": score,
            "model_state_dict": {"weight": torch.tensor([score])},
        }

    def test_atomic_save_replaces_best_and_keeps_only_best_pth(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            torch.save(self.checkpoint(1, 0.70), checkpoint_dir / "1-test0.70.pth")
            atomic_save_best(self.checkpoint(2, 0.80), checkpoint_dir)

            files = list(checkpoint_dir.glob("*.pth"))
            self.assertEqual([path.name for path in files], ["best.pth"])
            saved = torch.load(files[0], map_location="cpu", weights_only=False)
            self.assertEqual(saved["epoch"], 2)

    def test_prepare_migrates_highest_valid_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            torch.save(self.checkpoint(3, 0.71), checkpoint_dir / "3-test0.71.pth")
            torch.save(self.checkpoint(8, 0.83), checkpoint_dir / "8-test0.83.pth")
            (checkpoint_dir / "broken.pth").write_bytes(b"not a checkpoint")

            selected = prepare_best_checkpoint(checkpoint_dir)

            self.assertEqual(selected, checkpoint_dir / "best.pth")
            self.assertEqual([path.name for path in checkpoint_dir.glob("*.pth")], ["best.pth"])
            saved = torch.load(selected, map_location="cpu", weights_only=False)
            self.assertEqual(saved["epoch"], 8)

    def test_prepare_returns_none_when_no_checkpoint_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(prepare_best_checkpoint(Path(directory)))


if __name__ == "__main__":
    unittest.main()
