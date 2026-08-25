import json
import subprocess
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from main_torch import set_training_stage
from one_seed_models import build_experiment_model


ROOT = Path(__file__).resolve().parents[1]
BOTTLENECK = "one_seed_03_baseline_plus_fafem"
PLACEMENTS = (
    ("one_seed_02_fafem_bottleneck_stage3", False, False, True),
    ("one_seed_03_fafem_bottleneck_stage3_stage2", False, True, True),
    ("one_seed_04_fafem_bottleneck_stage3_stage2_stage1", True, True, True),
)


class FAFEMPlacementAblationTests(unittest.TestCase):
    def test_registry_is_cumulative_and_other_settings_match_bottleneck(self):
        reference = get_experiment(BOTTLENECK).to_dict()
        reference.pop("name")
        for name, stage1, stage2, stage3 in PLACEMENTS:
            config = get_experiment(name)
            self.assertTrue(config.enable_fafem)
            self.assertEqual(
                (config.fafem_stage1, config.fafem_stage2, config.fafem_stage3),
                (stage1, stage2, stage3),
            )
            candidate = config.to_dict()
            candidate.pop("name")
            candidate.pop("fafem_stage1", None)
            candidate.pop("fafem_stage2", None)
            candidate.pop("fafem_stage3", None)
            self.assertEqual(candidate, reference)

    def test_detected_stage_channels_and_parameter_counts(self):
        expected = {
            "one_seed_02_fafem_bottleneck_stage3": (217802, 29407900),
            "one_seed_03_fafem_bottleneck_stage3_stage2": (232599, 29422697),
            "one_seed_04_fafem_bottleneck_stage3_stage2_stage1": (238464, 29428562),
        }
        for name, _, _, _ in PLACEMENTS:
            model = build_experiment_model(get_experiment(name), None)
            active_modules = [
                module for module in (
                    model.fafem,
                    model.fafem_stage1,
                    model.fafem_stage2,
                    model.fafem_stage3,
                ) if module is not None
            ]
            fafem_parameters = sum(
                parameter.numel()
                for module in active_modules
                for parameter in module.parameters()
            )
            total_parameters = sum(parameter.numel() for parameter in model.parameters())
            self.assertEqual((fafem_parameters, total_parameters), expected[name])

    def test_bottleneck_only_state_and_initialization_are_unchanged(self):
        torch.manual_seed(42)
        reference = build_experiment_model(get_experiment(BOTTLENECK), None)
        self.assertIsNone(reference.fafem_stage1)
        self.assertIsNone(reference.fafem_stage2)
        self.assertIsNone(reference.fafem_stage3)
        reference_state = reference.state_dict()
        self.assertTrue(any(key.startswith("fafem.") for key in reference_state))
        self.assertFalse(any(key.startswith("fafem_stage") for key in reference_state))

        torch.manual_seed(42)
        cumulative = build_experiment_model(get_experiment(PLACEMENTS[-1][0]), None)
        cumulative_state = cumulative.state_dict()
        for key, value in reference_state.items():
            self.assertTrue(torch.equal(value, cumulative_state[key]), key)

    def test_active_fafem_modules_train_during_decoder_stage(self):
        model = build_experiment_model(get_experiment(PLACEMENTS[-1][0]), None)
        set_training_stage(model, "decoder")
        for prefix in ("fafem.", "fafem_stage1.", "fafem_stage2.", "fafem_stage3."):
            parameters = [
                parameter for name, parameter in model.named_parameters()
                if name.startswith(prefix)
            ]
            self.assertTrue(parameters, prefix)
            self.assertTrue(all(parameter.requires_grad for parameter in parameters), prefix)
        self.assertFalse(any(parameter.requires_grad for parameter in model.encoder.parameters()))

    def test_each_enhanced_skip_reaches_its_decoder_fusion(self):
        model = build_experiment_model(get_experiment(PLACEMENTS[-1][0]), None).eval()
        enhanced = {}
        fused = {}
        handles = []
        stage_modules = {
            "stage1": (model.fafem_stage1, model.bsei2, 96),
            "stage2": (model.fafem_stage2, model.bsei3, 192),
            "stage3": (model.fafem_stage3, model.bsei4, 384),
        }
        for name, (fafem, fusion, channels) in stage_modules.items():
            handles.append(fafem.register_forward_hook(
                lambda _module, _inputs, output, key=name:
                    enhanced.__setitem__(key, output.detach().clone())
            ))
            handles.append(fusion.register_forward_pre_hook(
                lambda _module, inputs, key=name, count=channels:
                    fused.__setitem__(key, inputs[0][:, -count:].detach().clone())
            ))
        try:
            with torch.no_grad():
                model(torch.randn(1, 3, 64, 64))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(set(enhanced), set(stage_modules))
        self.assertEqual(set(fused), set(stage_modules))
        for name in stage_modules:
            self.assertTrue(torch.equal(enhanced[name], fused[name]), name)

    def test_all_placement_variants_preserve_352_output_shape(self):
        for name in (BOTTLENECK,) + tuple(item[0] for item in PLACEMENTS):
            model = build_experiment_model(get_experiment(name), None).eval()
            with torch.no_grad():
                output = model(torch.randn(1, 3, 352, 352))
            self.assertEqual(output.shape, (1, 1, 352, 352), name)

    def test_runners_are_fixed_to_seed_42_and_isolated(self):
        runners = (
            "02_fafem_bottleneck_stage3",
            "03_fafem_bottleneck_stage3_stage2",
            "04_fafem_bottleneck_stage3_stage2_stage1",
        )
        expected_flags = (
            ("False", "False", "True"),
            ("False", "True", "True"),
            ("True", "True", "True"),
        )
        for runner, flags in zip(runners, expected_flags):
            completed = subprocess.run(
                [
                    "powershell", "-NoProfile", "-File",
                    str(ROOT / "ps_one_seed_ablation" / "FAFEM_ablation" /
                        f"{runner}.ps1"),
                    "-DryRun",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            train = payload["train"]
            self.assertEqual(payload["experiment"], f"one_seed_{runner}")
            self.assertEqual(payload["output_name"], runner)
            self.assertTrue(payload["seed_dir"].endswith(f"{runner}\\seed_42"))
            self.assertEqual(train[train.index("--seed") + 1], "42")
            self.assertEqual(train[train.index("--enable_fafem") + 1], "True")
            for option, expected in zip(
                ("--fafem_stage1", "--fafem_stage2", "--fafem_stage3"), flags
            ):
                self.assertEqual(train[train.index(option) + 1], expected)
            encoder_weights = train[train.index("--encoder_weights") + 1]
            self.assertTrue(encoder_weights.endswith("convnext_tiny_22k_1k_384.pth"))


if __name__ == "__main__":
    unittest.main()
