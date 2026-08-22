import json
import tempfile
import unittest
from pathlib import Path

from ablation_artifacts import (
    DATASETS,
    METRICS,
    atomic_write_json,
    validate_evaluation_summary,
)


class AblationArtifactTests(unittest.TestCase):
    def payload(self):
        result = {metric: 0.5 for metric in METRICS}
        return {
            "experiment_name": "01_baseline",
            "seed": 42,
            "checkpoint": "best_checkpoint.pth",
            "checkpoint_fingerprint": "abc",
            "trainable_parameters": 10,
            "total_parameters": 20,
            "macs": 100,
            "flops": 200,
            "gmacs": 1e-7,
            "gflops": 2e-7,
            "results": {dataset: dict(result) for dataset in DATASETS},
        }

    def test_valid_summary_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation_summary.json"
            atomic_write_json(path, self.payload())
            result = validate_evaluation_summary(path, "01_baseline", 42, "abc")
            self.assertTrue(result.valid, result.reason)

    def test_incomplete_or_wrong_fingerprint_summary_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation_summary.json"
            payload = self.payload()
            del payload["results"][DATASETS[0]][METRICS[0]]
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(validate_evaluation_summary(path, "01_baseline", 42, "abc").valid)
            payload = self.payload()
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(validate_evaluation_summary(path, "01_baseline", 42, "different").valid)

    def test_malformed_json_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation_summary.json"
            path.write_text("{", encoding="utf-8")
            self.assertFalse(validate_evaluation_summary(path, "01_baseline", 42, "abc").valid)


if __name__ == "__main__":
    unittest.main()
