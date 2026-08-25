import json
import subprocess
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from one_seed_models import build_experiment_model


ROOT = Path(__file__).resolve().parents[1]


class CSAFFAFEMAblationTests(unittest.TestCase):
    def test_registry_differs_from_baseline_only_by_two_modules(self):
        baseline = get_experiment("one_seed_01_baseline").to_dict()
        combined = get_experiment(
            "one_seed_04_baseline_plus_csaf_fafem"
        ).to_dict()
        self.assertIs(combined.pop("enable_csaf"), True)
        self.assertIs(combined.pop("enable_fafem"), True)
        baseline.pop("name")
        combined.pop("name")
        self.assertEqual(combined, baseline)

    def test_combined_model_contains_both_modules_and_preserves_output_shape(self):
        model = build_experiment_model(
            get_experiment("one_seed_04_baseline_plus_csaf_fafem"),
            encoder_weights=None,
        ).eval()
        self.assertIsNotNone(model.csaf)
        self.assertIsNotNone(model.fafem)
        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(output.shape, (1, 1, 64, 64))

    def test_runner_enables_only_csaf_and_fafem(self):
        completed = subprocess.run(
            [
                "powershell", "-NoProfile", "-File",
                str(ROOT / "ps_one_seed_ablation" / "04_baseline_plus_csaf_fafem.ps1"),
                "-DryRun",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        train = payload["train"]
        self.assertEqual(
            payload["experiment"], "one_seed_04_baseline_plus_csaf_fafem"
        )
        self.assertEqual(payload["output_name"], "04_baseline_plus_csaf_fafem")
        expected = {
            "--seed": "42",
            "--batch_size": "24",
            "--enable_csaf": "True",
            "--enable_fafem": "True",
            "--enable_msc": "False",
            "--skip_mode": "normal",
            "--detail_channels": "0",
            "--enable_gdf": "False",
            "--deep_supervision_heads": "0",
            "--decoder_warmup_epochs": "15",
            "--unfreeze_plateau_patience": "8",
            "--lr_plateau_patience": "12",
            "--early_stop_patience": "30",
        }
        for option, value in expected.items():
            self.assertEqual(train[train.index(option) + 1], value)
        self.assertIn("04_baseline_plus_csaf_fafem", payload["seed_dir"])


if __name__ == "__main__":
    unittest.main()
