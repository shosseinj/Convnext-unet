import json
import subprocess
import unittest
from pathlib import Path

import torch
from torch import nn

from ablation_registry import get_experiment
from models.convnext_pretrain import (
    DeformableDepthwiseMSCBBranch,
    PartialDeformableResidualFrequencyGuidedMSCBLite,
    ResidualFrequencyGuidedMSCBLite,
)
from one_seed_models import build_experiment_model


ROOT = Path(__file__).resolve().parents[1]
EXP45 = "one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine"
EXP50 = "one_seed_50_fafem_residual_frequency_guided_partial_deformable_mscb_stage3_warmup_cosine"
RUNNER = ROOT / "ps_one_seed_ablation" / "50_fafem_residual_frequency_guided_partial_deformable_mscb_stage3_warmup_cosine_seed42.ps1"


class PartialDeformableResidualFGMSCBTests(unittest.TestCase):
    def test_registry_delta_is_only_partial_deformable_branch_replacement(self):
        reference = get_experiment(EXP45).to_dict()
        candidate = get_experiment(EXP50).to_dict()
        self.assertTrue(candidate.pop("enable_partial_deformable_residual_fg_mscb_lite_stage3"))
        reference.pop("enable_residual_fg_mscb_lite_stage3")
        for config in (reference, candidate):
            config.pop("name")
        self.assertEqual(reference, candidate)

    def test_branch_types_and_zero_offset_equivalence(self):
        ordinary = ResidualFrequencyGuidedMSCBLite(384, 1536).eval()
        partial = PartialDeformableResidualFrequencyGuidedMSCBLite(384, 1536).eval()
        self.assertIsInstance(partial.dwconvs[0], nn.Sequential)
        self.assertIsInstance(partial.dwconvs[1], DeformableDepthwiseMSCBBranch)
        self.assertIsInstance(partial.dwconvs[2], DeformableDepthwiseMSCBBranch)
        data = torch.randn(1, 768, 22, 22)
        for index in (1, 2):
            partial.dwconvs[index].weight.data.copy_(ordinary.dwconvs[index][0].weight.data)
            partial.dwconvs[index].norm.load_state_dict(ordinary.dwconvs[index][1].state_dict())
            with torch.no_grad():
                self.assertTrue(torch.allclose(
                    partial.dwconvs[index](data), ordinary.dwconvs[index](data),
                    atol=2e-5, rtol=2e-4,
                ))

    def test_model_shape_alpha_and_gradients(self):
        model = build_experiment_model(get_experiment(EXP50), encoder_weights=None)
        module = model.partial_deformable_residual_fg_mscb_lite_stage3
        self.assertIsNotNone(module)
        self.assertIsNone(model.residual_fg_mscb_lite_stage3)
        features = torch.randn(2, 384, 22, 22)
        descriptor = torch.randn(2, 1536)
        output = module(features, descriptor)
        self.assertEqual(output.shape, features.shape)
        self.assertTrue(torch.allclose(module.branch_weights(descriptor).sum(dim=1), torch.full((2,), 3.0)))
        output.square().mean().backward()
        self.assertIsNotNone(module.dwconvs[1].offset.weight.grad)
        self.assertIsNotNone(module.dwconvs[2].offset.weight.grad)
        self.assertIsNotNone(module.guidance_strength.grad)

    def test_exp45_remains_ordinary_and_runner_dry_run_is_isolated(self):
        reference = build_experiment_model(get_experiment(EXP45), encoder_weights=None)
        self.assertIsInstance(reference.residual_fg_mscb_lite_stage3, ResidualFrequencyGuidedMSCBLite)
        self.assertFalse(hasattr(reference, "partial_deformable_residual_fg_mscb_lite_stage3") and
                         reference.partial_deformable_residual_fg_mscb_lite_stage3 is not None)
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(RUNNER), "-DryRun"],
            cwd=ROOT, text=True, capture_output=True, check=True,
        )
        command = dict(zip(json.loads(result.stdout.strip())["train"][::2],
                           json.loads(result.stdout.strip())["train"][1::2]))
        self.assertEqual(command["--experiment_name"], EXP50)
        self.assertEqual(command["--enable_partial_deformable_residual_fg_mscb_lite_stage3"], "True")


if __name__ == "__main__":
    unittest.main()
