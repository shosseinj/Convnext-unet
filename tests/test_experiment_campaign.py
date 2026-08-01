import json
import tempfile
import unittest
from pathlib import Path

from tools.run_experiment_campaign import valid_deep_report


class ExperimentCampaignTests(unittest.TestCase):
    def test_deep_report_identity_and_status_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.assertFalse(valid_deep_report(run_dir, "baseline", 42))
            path = run_dir / "validation.json"
            path.write_text(json.dumps({"status": "FAIL", "variant": "baseline", "seed": 42}), encoding="utf-8")
            self.assertFalse(valid_deep_report(run_dir, "baseline", 42))
            path.write_text(json.dumps({"status": "PASS", "variant": "baseline", "seed": 42}), encoding="utf-8")
            self.assertTrue(valid_deep_report(run_dir, "baseline", 42))
            self.assertFalse(valid_deep_report(run_dir, "baseline", 3407))


if __name__ == "__main__":
    unittest.main()
