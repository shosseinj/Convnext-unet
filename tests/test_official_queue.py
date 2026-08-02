import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.run_official_queue import (acquire_lock, deep_validation_pass,
                                      load_incremental_jobs, training_command, validate_completed_run,
                                      write_run_status)


class OfficialQueueTests(unittest.TestCase):
    def test_queue_is_derived_from_canonical_matrix(self):
        jobs = load_incremental_jobs("configs/ablation_matrix.yaml")
        self.assertEqual(len(jobs), 18)
        self.assertEqual(jobs[0], ("baseline", 42))
        self.assertEqual(jobs[-1], ("full", 2026))

    def make_run(self, root):
        run = Path(root)
        summary = {"status": "PASS", "completed": True, "variant": "baseline",
                   "seed": 42, "epochs_completed": 1}
        config = {"variant": "baseline", "seed": 42, "run_mode": "official",
                  "environment": {}, "model": {}, "source_fingerprints": {},
                  "pretrained_weights": {}}
        history = [{"train_loss": 1.0, "selection_dice": 0.5}]
        for name, value in (("summary.json", summary), ("resolved_config.json", config),
                            ("history.json", history)):
            (run / name).write_text(json.dumps(value), encoding="utf-8")
        (run / "best.pth").write_bytes(b"checkpoint")
        (run / "last.pth").write_bytes(b"checkpoint")
        return run

    def test_completed_run_requires_complete_finite_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(directory)
            self.assertEqual(validate_completed_run(run, "baseline", 42),
                             (True, "validated completed run"))
            history = [{"train_loss": float("nan"), "selection_dice": 0.5}]
            (run / "history.json").write_text(json.dumps(history), encoding="utf-8")
            valid, reason = validate_completed_run(run, "baseline", 42)
            self.assertFalse(valid)
            self.assertIn("non-finite", reason)

    def test_lock_rejects_live_owner_and_replaces_stale_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.lock.json"
            path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "already active"):
                acquire_lock(path)
            path.write_text(json.dumps({"pid": 99999999}), encoding="utf-8")
            owned = acquire_lock(path)
            self.assertEqual(owned["pid"], os.getpid())

    def test_deep_validation_identity_and_short_status_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root / "run"; run.mkdir()
            (run / "validation.json").write_text(json.dumps({
                "status": "PASS", "variant": "baseline", "seed": 42}), encoding="utf-8")
            self.assertTrue(deep_validation_pass(run, "baseline", 42))
            self.assertFalse(deep_validation_pass(run, "baseline", 3407))
            write_run_status(root, campaign_status="RUNNING", active_run="baseline / seed 42",
                             latest_epoch="Starting", completed=0, last_completed="None",
                             last_result="Run started", process_status="Visible terminal active",
                             last_error="None", next_action="Train")
            lines = (root / "RUN_STATUS.md").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 12)
            self.assertEqual(lines[0], "Stage: EXPERIMENT")

    def test_resume_checkpoint_uses_cpu_load_wrapper_without_changing_training_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); script = root / "train_research.py"; script.write_text("pass")
            checkpoint = root / "last.pth"
            fresh = training_command("python", script, "baseline", 2026, "cuda", checkpoint)
            self.assertNotIn("-c", fresh)
            checkpoint.write_bytes(b"checkpoint")
            resumed = training_command("python", script, "baseline", 2026, "cuda", checkpoint)
            self.assertEqual(resumed[1:3], ["-u", "-c"])
            self.assertIn("map_location='cpu'", resumed[3])
            self.assertEqual(resumed[4], str(script))


if __name__ == "__main__":
    unittest.main()
