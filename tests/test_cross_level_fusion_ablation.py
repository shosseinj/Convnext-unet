import json
import subprocess
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from main_torch import set_training_stage
from models.cross_level_fusion import CrossLevelFusion
from one_seed_models import build_experiment_model


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "one_seed_03_baseline_plus_fafem"
EXPERIMENT = "one_seed_05_fafem_plus_cross_level_fusion"


class CrossLevelFusionAblationTests(unittest.TestCase):
    def test_registry_differs_from_fafem_only_by_name_and_fusion_flag(self):
        reference = get_experiment(REFERENCE).to_dict()
        candidate = get_experiment(EXPERIMENT).to_dict()
        self.assertTrue(candidate.pop("enable_cross_level_fusion"))
        reference.pop("name")
        candidate.pop("name")
        self.assertEqual(candidate, reference)

    def test_module_preserves_skip_shapes_and_starts_as_identity(self):
        module = CrossLevelFusion((96, 192, 384), fusion_channels=96)
        features = (
            torch.randn(1, 96, 88, 88),
            torch.randn(1, 192, 44, 44),
            torch.randn(1, 384, 22, 22),
        )
        with torch.no_grad():
            outputs = module(*features)
        for original, output in zip(features, outputs):
            self.assertEqual(output.shape, original.shape)
            self.assertTrue(torch.equal(output, original))

    def test_parameter_counts_are_lightweight_and_exact(self):
        reference = build_experiment_model(get_experiment(REFERENCE), None)
        model = build_experiment_model(get_experiment(EXPERIMENT), None)
        fusion_parameters = sum(
            parameter.numel() for parameter in model.cross_level_fusion.parameters()
        )
        self.assertEqual(fusion_parameters, 133890)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            sum(parameter.numel() for parameter in reference.parameters()) + 133890,
        )

    def test_fafem_only_state_and_initialization_remain_unchanged(self):
        torch.manual_seed(42)
        reference = build_experiment_model(get_experiment(REFERENCE), None)
        self.assertIsNone(reference.cross_level_fusion)
        reference_state = reference.state_dict()
        self.assertFalse(any(key.startswith("cross_level_fusion.") for key in reference_state))

        torch.manual_seed(42)
        model = build_experiment_model(get_experiment(EXPERIMENT), None)
        model_state = model.state_dict()
        for key, value in reference_state.items():
            self.assertTrue(torch.equal(value, model_state[key]), key)

    def test_fusion_is_trainable_during_encoder_frozen_stage(self):
        model = build_experiment_model(get_experiment(EXPERIMENT), None)
        set_training_stage(model, "decoder")
        fusion_parameters = list(model.cross_level_fusion.parameters())
        self.assertTrue(fusion_parameters)
        self.assertTrue(all(parameter.requires_grad for parameter in fusion_parameters))
        self.assertFalse(any(parameter.requires_grad for parameter in model.encoder.parameters()))

    def test_refined_skips_reach_expected_decoder_fusions(self):
        model = build_experiment_model(get_experiment(EXPERIMENT), None).eval()
        refined = {}
        consumed = {}
        handles = [
            model.cross_level_fusion.register_forward_hook(
                lambda _module, _inputs, outputs: refined.update(
                    {"stage1": outputs[0].detach().clone(),
                     "stage2": outputs[1].detach().clone(),
                     "stage3": outputs[2].detach().clone()}
                )
            ),
            model.bsei2.register_forward_pre_hook(
                lambda _module, inputs: consumed.__setitem__(
                    "stage1", inputs[0][:, -96:].detach().clone()
                )
            ),
            model.bsei3.register_forward_pre_hook(
                lambda _module, inputs: consumed.__setitem__(
                    "stage2", inputs[0][:, -192:].detach().clone()
                )
            ),
            model.bsei4.register_forward_pre_hook(
                lambda _module, inputs: consumed.__setitem__(
                    "stage3", inputs[0][:, -384:].detach().clone()
                )
            ),
        ]
        try:
            with torch.no_grad():
                output = model(torch.randn(1, 3, 352, 352))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(output.shape, (1, 1, 352, 352))
        self.assertEqual(set(refined), {"stage1", "stage2", "stage3"})
        self.assertEqual(set(consumed), set(refined))
        for stage in refined:
            self.assertEqual(refined[stage].shape, consumed[stage].shape)
            self.assertTrue(torch.equal(refined[stage], consumed[stage]), stage)

    def test_three_seed_runner_is_isolated_and_protocol_matched(self):
        completed = subprocess.run(
            [
                "powershell", "-NoProfile", "-File",
                str(ROOT / "ps_one_seed_ablation" /
                    "05_fafem_plus_cross_level_fusion.ps1"),
                "-DryRun",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payloads = [
            json.loads(line) for line in completed.stdout.splitlines()
            if line.strip().startswith("{")
        ]
        self.assertEqual([payload["train"][payload["train"].index("--seed") + 1]
                          for payload in payloads], ["42", "7777", "6543"])
        for payload in payloads:
            train = payload["train"]
            seed = train[train.index("--seed") + 1]
            self.assertEqual(payload["experiment"], EXPERIMENT)
            self.assertEqual(payload["output_name"], "05_fafem_plus_cross_level_fusion")
            self.assertTrue(payload["seed_dir"].endswith(f"seed_{seed}"))
            expected = {
                "--enable_msc": "False",
                "--skip_mode": "normal",
                "--detail_channels": "0",
                "--enable_gdf": "False",
                "--enable_csaf": "False",
                "--enable_fafem": "True",
                "--fafem_stage1": "False",
                "--fafem_stage2": "False",
                "--fafem_stage3": "False",
                "--enable_cross_level_fusion": "True",
                "--deep_supervision_heads": "0",
                "--batch_size": "24",
                "--decoder_warmup_epochs": "15",
                "--unfreeze_plateau_patience": "8",
                "--lr_plateau_patience": "12",
                "--early_stop_patience": "30",
            }
            for option, value in expected.items():
                self.assertEqual(train[train.index(option) + 1], value)


if __name__ == "__main__":
    unittest.main()
