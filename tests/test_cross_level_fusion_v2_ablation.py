import json
import subprocess
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from main_torch import set_training_stage
from models.cross_level_fusion import CrossLevelFusion, CrossLevelFusionV2
from one_seed_models import build_experiment_model


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "one_seed_03_baseline_plus_fafem"
V1 = "one_seed_05_fafem_plus_cross_level_fusion"
V2 = "one_seed_06_fafem_plus_cross_level_fusion_v2"


class CrossLevelFusionV2AblationTests(unittest.TestCase):
    def test_registry_differs_from_v1_only_by_name_and_version(self):
        v1 = get_experiment(V1).to_dict()
        v2 = get_experiment(V2).to_dict()
        self.assertEqual(v2.pop("cross_level_fusion_version"), "v2")
        v1.pop("name")
        v2.pop("name")
        self.assertEqual(v2, v1)

    def test_v1_implementation_and_parameter_count_are_unchanged(self):
        model = build_experiment_model(get_experiment(V1), None)
        self.assertIsInstance(model.cross_level_fusion, CrossLevelFusion)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.cross_level_fusion.parameters()),
            133890,
        )

    def test_v2_uses_adjacent_sources_and_preserves_native_shapes(self):
        module = CrossLevelFusionV2((96, 192, 384), fusion_channels=64)
        self.assertEqual(module.context_sources, ((0, 1), (0, 1, 2), (1, 2)))
        features = (
            torch.randn(1, 96, 88, 88),
            torch.randn(1, 192, 44, 44),
            torch.randn(1, 384, 22, 22),
        )
        with torch.no_grad():
            outputs = module(*features)
        self.assertEqual(
            tuple(output.shape for output in outputs),
            tuple(feature.shape for feature in features),
        )
        self.assertTrue(torch.allclose(
            module.residual_scales.detach(), torch.full((3,), 1e-3)
        ))

    def test_nonzero_layerscale_allows_immediate_internal_gradients(self):
        module = CrossLevelFusionV2((8, 16, 32), fusion_channels=8)
        outputs = module(
            torch.randn(2, 8, 16, 16),
            torch.randn(2, 16, 8, 8),
            torch.randn(2, 32, 4, 4),
        )
        sum(output.square().mean() for output in outputs).backward()
        for projection in module.input_projections:
            gradient = projection[0].weight.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_v2_parameter_and_total_counts_are_exact(self):
        reference = build_experiment_model(get_experiment(REFERENCE), None)
        model = build_experiment_model(get_experiment(V2), None)
        fusion_parameters = sum(
            parameter.numel() for parameter in model.cross_level_fusion.parameters()
        )
        self.assertEqual(fusion_parameters, 93156)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            sum(parameter.numel() for parameter in reference.parameters()) + 93156,
        )

    def test_fafem_only_common_initialization_remains_unchanged(self):
        torch.manual_seed(42)
        reference = build_experiment_model(get_experiment(REFERENCE), None)
        torch.manual_seed(42)
        model = build_experiment_model(get_experiment(V2), None)
        model_state = model.state_dict()
        for key, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, model_state[key]), key)

    def test_v2_is_trainable_with_frozen_encoder(self):
        model = build_experiment_model(get_experiment(V2), None)
        set_training_stage(model, "decoder")
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in model.cross_level_fusion.parameters()
        ))
        self.assertFalse(any(
            parameter.requires_grad for parameter in model.encoder.parameters()
        ))

    def test_full_model_preserves_output_and_decoder_skip_shapes(self):
        model = build_experiment_model(get_experiment(V2), None).eval()
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
        for stage in refined:
            self.assertEqual(refined[stage].shape, consumed[stage].shape)
            self.assertTrue(torch.equal(refined[stage], consumed[stage]), stage)

    def test_v2_runner_has_three_isolated_seed_commands(self):
        completed = subprocess.run(
            [
                "powershell", "-NoProfile", "-File",
                str(ROOT / "ps_one_seed_ablation" /
                    "06_fafem_plus_cross_level_fusion_v2.ps1"),
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
        seeds = [
            payload["train"][payload["train"].index("--seed") + 1]
            for payload in payloads
        ]
        self.assertEqual(seeds, ["42", "7777", "6543"])
        for payload, seed in zip(payloads, seeds):
            train = payload["train"]
            self.assertEqual(payload["experiment"], V2)
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
                "--cross_level_fusion_version": "v2",
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
