import json
import subprocess
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from models.fafem import FrequencyAwareFeatureEnhancement
from one_seed_models import build_experiment_model


ROOT = Path(__file__).resolve().parents[1]


class FAFEMAblationTests(unittest.TestCase):
    def test_registry_differs_from_baseline_only_by_name_and_fafem(self):
        baseline = get_experiment("one_seed_01_baseline").to_dict()
        fafem = get_experiment("one_seed_03_baseline_plus_fafem").to_dict()
        self.assertNotIn("enable_fafem", baseline)
        self.assertNotIn("enable_csaf", fafem)
        self.assertIs(fafem.pop("enable_fafem"), True)
        baseline.pop("name")
        fafem.pop("name")
        self.assertEqual(fafem, baseline)

    def test_fafem_preserves_shape_and_starts_as_residual_identity(self):
        feature = torch.randn(2, 32, 11, 11)
        module = FrequencyAwareFeatureEnhancement(32)
        output = module(feature)
        self.assertEqual(output.shape, feature.shape)
        self.assertTrue(torch.equal(output, feature))

    def test_fafem_does_not_change_common_baseline_initialization(self):
        torch.manual_seed(42)
        baseline = build_experiment_model(
            get_experiment("one_seed_01_baseline"), encoder_weights=None
        )
        torch.manual_seed(42)
        fafem = build_experiment_model(
            get_experiment("one_seed_03_baseline_plus_fafem"), encoder_weights=None
        )
        baseline_state = baseline.state_dict()
        fafem_state = fafem.state_dict()
        self.assertFalse(any(key.startswith("fafem.") for key in baseline_state))
        self.assertTrue(any(key.startswith("fafem.") for key in fafem_state))
        for key, value in baseline_state.items():
            self.assertTrue(torch.equal(value, fafem_state[key]), key)

    def test_full_model_preserves_352_segmentation_shape(self):
        model = build_experiment_model(
            get_experiment("one_seed_03_baseline_plus_fafem"),
            encoder_weights=None,
        ).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 352, 352))
        self.assertEqual(output.shape, (1, 1, 352, 352))

    def test_runner_is_isolated_and_reuses_baseline_protocol(self):
        completed = subprocess.run(
            [
                "powershell", "-NoProfile", "-File",
                str(ROOT / "ps_one_seed_ablation" / "03_baseline_plus_fafem.ps1"),
                "-DryRun",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        train = payload["train"]
        self.assertEqual(payload["experiment"], "one_seed_03_baseline_plus_fafem")
        self.assertEqual(payload["output_name"], "03_baseline_plus_fafem")
        self.assertIn("03_baseline_plus_fafem", payload["seed_dir"])
        expected = {
            "--seed": "42",
            "--batch_size": "24",
            "--enable_msc": "False",
            "--skip_mode": "normal",
            "--detail_channels": "0",
            "--enable_gdf": "False",
            "--enable_csaf": "False",
            "--enable_fafem": "True",
            "--deep_supervision_heads": "0",
            "--decoder_warmup_epochs": "15",
            "--unfreeze_plateau_patience": "8",
            "--lr_plateau_patience": "12",
            "--early_stop_patience": "30",
        }
        for option, value in expected.items():
            self.assertEqual(train[train.index(option) + 1], value)


if __name__ == "__main__":
    unittest.main()
