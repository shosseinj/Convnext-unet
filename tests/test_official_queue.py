import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.run_official_queue import acquire_lock, load_incremental_jobs, validate_completed_run


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


if __name__ == "__main__":
    unittest.main()
