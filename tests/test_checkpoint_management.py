import tempfile
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from checkpoint_management import (
    atomic_save_best,
    mark_training_complete,
    prepare_best_checkpoint,
    prepare_checkpoint,
)


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
            selected = prepare_best_checkpoint(checkpoint_dir)

            self.assertEqual(selected, checkpoint_dir / "best.pth")
            self.assertEqual([path.name for path in checkpoint_dir.glob("*.pth")], ["best.pth"])
            saved = torch.load(selected, map_location="cpu", weights_only=False)
            self.assertEqual(saved["epoch"], 8)

    def test_corrupt_canonical_checkpoint_is_error(self):
        with tempfile.TemporaryDirectory() as directory:
            seed_dir = Path(directory)
            (seed_dir / "best_checkpoint.pth").write_bytes(b"broken")
            decision = prepare_checkpoint(
                seed_dir, get_experiment("01_baseline"), 42, 150, seed_dir / "log.txt"
            )
            self.assertEqual(decision.action, "error")
            self.assertIn("cannot be loaded", decision.reason)

    def test_valid_incomplete_checkpoint_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            seed_dir = Path(directory)
            checkpoint = self.checkpoint(10, 0.8)
            checkpoint.update({
                "experiment_name": "01_baseline",
                "seed": 42,
                "architecture": get_experiment("01_baseline").to_dict(),
                "training_complete": False,
            })
            torch.save(checkpoint, seed_dir / "best_checkpoint.pth")
            decision = prepare_checkpoint(
                seed_dir, get_experiment("01_baseline"), 42, 150, seed_dir / "log.txt"
            )
            self.assertEqual(decision.action, "resume")

    def test_completed_legacy_checkpoint_is_migrated_to_canonical_path(self):
        with tempfile.TemporaryDirectory() as directory:
            seed_dir = Path(directory)
            legacy_dir = seed_dir / "checkpoints_KvasirSEG-ConvNeXt"
            legacy_dir.mkdir()
            torch.save(self.checkpoint(133, 0.87), legacy_dir / "best.pth")
            log_path = seed_dir / "KvasirSEG-ConvNeXt_log.txt"
            log_path.write_text(
                "Epoch 150/150: Train Loss: 0.2\n"
                "Training stopped after epoch 150; best.pth retained.\n",
                encoding="utf-8",
            )

            config = get_experiment("01_baseline")
            decision = prepare_checkpoint(seed_dir, config, 42, 150, log_path)

            self.assertEqual(decision.action, "skip")
            self.assertEqual(decision.checkpoint_path, seed_dir / "best_checkpoint.pth")
            migrated = torch.load(decision.checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(migrated["experiment_name"], "01_baseline")
            self.assertEqual(migrated["seed"], 42)
            self.assertEqual(migrated["architecture"], config.to_dict())
            self.assertTrue(migrated["training_complete"])
            self.assertEqual(migrated["final_epoch"], 149)

    def test_incomplete_legacy_checkpoint_is_migrated_for_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            seed_dir = Path(directory)
            legacy_dir = seed_dir / "checkpoints_KvasirSEG-ConvNeXt"
            legacy_dir.mkdir()
            torch.save(self.checkpoint(20, 0.70), legacy_dir / "best.pth")
            log_path = seed_dir / "KvasirSEG-ConvNeXt_log.txt"
            log_path.write_text("Epoch 21/150: Train Loss: 0.4\n", encoding="utf-8")

            decision = prepare_checkpoint(
                seed_dir, get_experiment("01_baseline"), 42, 150, log_path
            )

            self.assertEqual(decision.action, "resume")
            migrated = torch.load(decision.checkpoint_path, map_location="cpu", weights_only=False)
            self.assertFalse(migrated["training_complete"])

    def test_mark_complete_preserves_best_weights(self):
        checkpoint = self.checkpoint(10, 0.8)
        before = checkpoint["model_state_dict"]["weight"].clone()
        completed = mark_training_complete(checkpoint, 149, 500.0, "max_epochs")
        torch.testing.assert_close(completed["model_state_dict"]["weight"], before)
        self.assertTrue(completed["training_complete"])
        self.assertEqual(completed["final_epoch"], 149)

    def test_prepare_returns_none_when_no_checkpoint_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(prepare_best_checkpoint(Path(directory)))


if __name__ == "__main__":
    unittest.main()
