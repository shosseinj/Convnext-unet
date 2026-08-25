import json
import subprocess
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from models.csaf import CrossScaleAttentionFusionV2
from one_seed_models import build_experiment_model


ROOT = Path(__file__).resolve().parents[1]


class CSAFV2AblationTests(unittest.TestCase):
    def test_registry_differs_from_fafem_only_by_csaf_v2(self):
        fafem = get_experiment("one_seed_03_baseline_plus_fafem").to_dict()
        v2 = get_experiment("one_seed_05_baseline_plus_fafem_csafv2").to_dict()
        self.assertIs(v2.pop("enable_csaf"), True)
        self.assertEqual(v2.pop("csaf_version"), "v2")
        fafem.pop("name")
        v2.pop("name")
        self.assertEqual(v2, fafem)

    def test_v2_is_adjacent_scale_identity_with_per_channel_gate(self):
        features = (
            torch.randn(2, 8, 32, 32),
            torch.randn(2, 16, 16, 16),
            torch.randn(2, 32, 8, 8),
            torch.randn(2, 64, 4, 4),
        )
        fusion = CrossScaleAttentionFusionV2(
            (8, 16, 32, 64), (0, 1, 2), 1, 16
        )
        output = fusion(features)
        self.assertEqual(fusion.gamma.shape, (1, 16, 1, 1))
        self.assertEqual(output.shape, features[1].shape)
        self.assertTrue(torch.equal(output, features[1]))

    def test_v2_preserves_all_common_fafem_initialization(self):
        torch.manual_seed(42)
        fafem = build_experiment_model(
            get_experiment("one_seed_03_baseline_plus_fafem"), None
        )
        torch.manual_seed(42)
        v2 = build_experiment_model(
            get_experiment("one_seed_05_baseline_plus_fafem_csafv2"), None
        )
        fafem_state = fafem.state_dict()
        v2_state = v2.state_dict()
        for key, value in fafem_state.items():
            self.assertTrue(torch.equal(value, v2_state[key]), key)

    def test_fafem_enhanced_f4_is_passed_into_csaf_v2(self):
        model = build_experiment_model(
            get_experiment("one_seed_05_baseline_plus_fafem_csafv2"), None
        ).eval()
        model.fafem.gamma.data.fill_(1.0)
        captured = {}

        def capture_fafem(_module, _inputs, output):
            captured["fafem_f4"] = output.detach().clone()

        def capture_csaf(_module, inputs):
            captured["csaf_f4"] = inputs[0][3].detach().clone()

        fafem_hook = model.fafem.register_forward_hook(capture_fafem)
        csaf_hook = model.csaf[2].register_forward_pre_hook(capture_csaf)
        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 64))
        fafem_hook.remove()
        csaf_hook.remove()
        self.assertEqual(output.shape, (1, 1, 64, 64))
        self.assertTrue(torch.equal(captured["fafem_f4"], captured["csaf_f4"]))

    def test_runner_is_isolated_and_uses_baseline_protocol(self):
        completed = subprocess.run(
            [
                "powershell", "-NoProfile", "-File",
                str(ROOT / "ps_one_seed_ablation" / "05_baseline_plus_fafem_csafv2.ps1"),
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
            payload["experiment"], "one_seed_05_baseline_plus_fafem_csafv2"
        )
        self.assertEqual(payload["output_name"], "05_baseline_plus_fafem_csafv2")
        expected = {
            "--seed": "42",
            "--batch_size": "24",
            "--enable_csaf": "True",
            "--enable_fafem": "True",
            "--csaf_version": "v2",
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
        self.assertIn("05_baseline_plus_fafem_csafv2", payload["seed_dir"])


if __name__ == "__main__":
    unittest.main()
