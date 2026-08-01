import json
import tempfile
import unittest
from pathlib import Path

import torch

from tools.validate_official_run import validate_deep


class OfficialRunValidationTests(unittest.TestCase):
    def build_run(self, root):
        root = Path(root)
        manifest = {"sha256": "manifest", "datasets": {
            "Kvasir-SEG": {"counts": {"validation": 1}},
            "CVC-ClinicDB": {"counts": {"validation": 1}},
        }}
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        metrics = {"dice": 0.8, "iou": 0.7, "soft_dice": 0.75,
                   "predicted_positive_fraction": 0.1, "mae": 0.05, "samples": 1}
        history = [{"epoch": 1, "train_loss": 1.0, "selection_dice": 0.8,
                    "validation": {"Kvasir-SEG": metrics, "CVC-ClinicDB": metrics}}]
        metadata = {"variant": "baseline", "seed": 42, "run_mode": "official",
                    "manifest": str(manifest_path), "manifest_sha256": "manifest",
                    "environment": {}, "model": {}, "source_fingerprints": {},
                    "pretrained_weights": {}}
        config = dict(metadata)
        summary = {"status": "PASS", "completed": True, "variant": "baseline", "seed": 42,
                   "epochs_completed": 1, "best_selection_dice": 0.8}
        for name, value in (("history.json", history), ("resolved_config.json", config),
                            ("summary.json", summary)):
            (root / name).write_text(json.dumps(value), encoding="utf-8")
        model = {"weight": torch.ones(1)}
        torch.save({"epoch": 0, "selection_dice": 0.8, "model_state_dict": model,
                    "metadata": metadata}, root / "best.pth")
        torch.save({"epoch": 0, "history": history, "best_score": 0.8,
                    "model_state_dict": model, "metadata": metadata, "rng_state": {},
                    "elapsed_seconds_total": 1.0}, root / "last.pth")
        return root

    def test_deep_validation_and_selection_formula_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.build_run(directory)
            report = validate_deep(run, "baseline", 42)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["recovery_contract"], "rng-restorable")
            history = json.loads((run / "history.json").read_text(encoding="utf-8"))
            history[0]["selection_dice"] = 0.9
            (run / "history.json").write_text(json.dumps(history), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Selection formula mismatch"):
                validate_deep(run, "baseline", 42)


if __name__ == "__main__":
    unittest.main()
