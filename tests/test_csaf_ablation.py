import json
import subprocess
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from models.csaf import CrossScaleAttentionFusion
from one_seed_models import build_experiment_model


ROOT = Path(__file__).resolve().parents[1]


class CSAFAblationTests(unittest.TestCase):
    def test_registry_differs_from_baseline_only_by_name_and_csaf(self):
        baseline = get_experiment("one_seed_01_baseline").to_dict()
        csaf = get_experiment("one_seed_02_baseline_plus_csaf").to_dict()
        self.assertNotIn("enable_csaf", baseline)
        self.assertIs(csaf.pop("enable_csaf"), True)
        baseline.pop("name")
        csaf.pop("name")
        self.assertEqual(csaf, baseline)

    def test_csaf_preserves_target_shape_and_starts_as_residual_identity(self):
        features = (
            torch.randn(2, 8, 32, 32),
            torch.randn(2, 16, 16, 16),
            torch.randn(2, 32, 8, 8),
            torch.randn(2, 64, 4, 4),
        )
        fusion = CrossScaleAttentionFusion((8, 16, 32, 64), 1, 16)
        output = fusion(features)
        self.assertEqual(output.shape, features[1].shape)
        self.assertTrue(torch.equal(output, features[1]))

    def test_baseline_has_no_csaf_parameters(self):
        baseline = build_experiment_model(
            get_experiment("one_seed_01_baseline"), encoder_weights=None
        )
        csaf = build_experiment_model(
            get_experiment("one_seed_02_baseline_plus_csaf"), encoder_weights=None
        )
        self.assertFalse(any(name.startswith("csaf.") for name, _ in baseline.named_parameters()))
        self.assertTrue(any(name.startswith("csaf.") for name, _ in csaf.named_parameters()))
        self.assertFalse(baseline.variant_config["enable_csaf"])
        self.assertTrue(csaf.variant_config["enable_csaf"])

    def test_runner_is_isolated_and_reuses_baseline_training_protocol(self):
        completed = subprocess.run(
            [
                "powershell", "-NoProfile", "-File",
                str(ROOT / "ps_one_seed_ablation" / "02_baseline_plus_csaf.ps1"),
                "-DryRun",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        train = payload["train"]
        self.assertEqual(payload["experiment"], "one_seed_02_baseline_plus_csaf")
        self.assertEqual(payload["output_name"], "02_baseline_plus_csaf")
        self.assertIn("02_baseline_plus_csaf", payload["seed_dir"])
        expected = {
            "--seed": "42",
            "--batch_size": "24",
            "--enable_msc": "False",
            "--skip_mode": "normal",
            "--detail_channels": "0",
            "--enable_gdf": "False",
            "--deep_supervision_heads": "0",
            "--enable_csaf": "True",
            "--decoder_warmup_epochs": "15",
            "--unfreeze_plateau_patience": "8",
            "--lr_plateau_patience": "12",
            "--early_stop_patience": "30",
        }
        for option, value in expected.items():
            self.assertEqual(train[train.index(option) + 1], value)
        self.assertNotIn("01_baseline\\seed_42", payload["seed_dir"])


if __name__ == "__main__":
    unittest.main()
