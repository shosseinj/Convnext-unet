import json
import subprocess
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from one_seed_models import build_experiment_model


EXP45 = "one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine"
EXP48 = "one_seed_48_fafem_residual_frequency_guided_mscb_all_skips_warmup_cosine"
ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "ps_one_seed_ablation" / "48_fafem_residual_frequency_guided_mscb_all_skips_warmup_cosine_seed42.ps1"


class ResidualFGMSCBAllSkipsTests(unittest.TestCase):
    def test_registry_differs_from_exp45_only_by_all_skip_fg_mscb(self):
        reference = get_experiment(EXP45).to_dict()
        candidate = get_experiment(EXP48).to_dict()
        reference.pop("enable_residual_fg_mscb_lite_stage3")
        candidate.pop("enable_residual_fg_mscb_all_skips")
        for config in (reference, candidate):
            config.pop("name")
        self.assertEqual(reference, candidate)

    def test_all_skip_modules_are_independent_and_preserve_shapes(self):
        model = build_experiment_model(get_experiment(EXP48), encoder_weights=None).eval()
        modules = (
            model.residual_fg_mscb_lite_stage1,
            model.residual_fg_mscb_lite_stage2,
            model.residual_fg_mscb_lite_stage3_skip,
        )
        self.assertTrue(all(module is not None for module in modules))
        self.assertEqual(len({id(module) for module in modules}), 3)
        descriptor = torch.randn(1, 1536)
        for module, channels, size in zip(modules, (96, 192, 384), (88, 44, 22)):
            feature = torch.randn(1, channels, size, size)
            self.assertEqual(module(feature, descriptor).shape, feature.shape)
            self.assertTrue(torch.allclose(module.branch_weights(descriptor).sum(dim=1), torch.tensor([3.0])))

    def test_all_skip_guidance_and_strength_parameters_receive_gradients(self):
        model = build_experiment_model(get_experiment(EXP48), encoder_weights=None)
        descriptor = torch.randn(2, 1536)
        modules = (
            (model.residual_fg_mscb_lite_stage1, 96, 88),
            (model.residual_fg_mscb_lite_stage2, 192, 44),
            (model.residual_fg_mscb_lite_stage3_skip, 384, 22),
        )
        loss = sum(module(torch.randn(2, channels, size, size), descriptor).square().mean()
                   for module, channels, size in modules)
        loss.backward()
        for module, _, _ in modules:
            self.assertGreater(module.guidance_mlp[-1].weight.grad.abs().sum().item(), 0.0)
            self.assertIsNotNone(module.guidance_strength.grad)

    def test_exp45_remains_post_fusion_stage3_only(self):
        model = build_experiment_model(get_experiment(EXP45), encoder_weights=None)
        self.assertIsNotNone(model.residual_fg_mscb_lite_stage3)
        self.assertIsNone(getattr(model, "residual_fg_mscb_lite_stage1", None))
        self.assertIsNone(getattr(model, "residual_fg_mscb_lite_stage2", None))
        self.assertIsNone(getattr(model, "residual_fg_mscb_lite_stage3_skip", None))

    def test_runner_dry_run(self):
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(RUNNER), "-DryRun"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        command = dict(zip(json.loads(completed.stdout.strip())["train"][::2],
                           json.loads(completed.stdout.strip())["train"][1::2]))
        self.assertEqual(command["--experiment_name"], EXP48)
        self.assertEqual(command["--enable_residual_fg_mscb_all_skips"], "True")
        self.assertEqual(command["--enable_residual_fg_mscb_lite_stage3"], "False")


if __name__ == "__main__":
    unittest.main()
