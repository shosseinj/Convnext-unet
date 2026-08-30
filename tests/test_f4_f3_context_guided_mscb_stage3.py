import json
import subprocess
import unittest
from pathlib import Path

import torch

from ablation_registry import get_experiment
from models.convnext_pretrain import F4F3ContextResidualFrequencyGuidedMSCBLite, ResidualFrequencyGuidedMSCBLite
from one_seed_models import build_experiment_model


ROOT = Path(__file__).resolve().parents[1]
EXP45 = "one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine"
EXP51 = "one_seed_51_fafem_f4_f3_context_guided_mscb_stage3_batch32"
RUNNER = ROOT / "ps_one_seed_ablation" / "51_fafem_f4_f3_context_guided_mscb_stage3_batch32_seed42.ps1"


class F4F3ContextGuidedMSCBTests(unittest.TestCase):
    def test_context_descriptors_alpha_and_gradients(self):
        module = F4F3ContextResidualFrequencyGuidedMSCBLite(384, 1536, initial_guidance_strength=0.05)
        f4_descriptor = torch.randn(2, 1536, requires_grad=True)
        f3 = torch.randn(2, 384, 22, 22, requires_grad=True)
        output = module(torch.randn(2, 384, 22, 22), f4_descriptor, f3)
        self.assertEqual(output.shape, (2, 384, 22, 22))
        self.assertEqual(module.last_descriptor_shapes, {"f4_frequency_descriptor": (2, 1536), "f3_local_descriptor": (2, 384), "combined_descriptor": (2, 1920)})
        alpha = module.last_alpha
        self.assertTrue(torch.equal(alpha, torch.ones_like(alpha)))
        self.assertTrue(torch.equal(alpha.sum(dim=1), torch.full((2,), 3.0)))
        output.square().mean().backward()
        self.assertIsNotNone(f4_descriptor.grad)
        self.assertIsNotNone(f3.grad)
        self.assertIsNotNone(module.guidance_strength.grad)

    def test_registry_exp45_regression_and_runner_dry_run(self):
        reference = get_experiment(EXP45).to_dict()
        candidate = get_experiment(EXP51).to_dict()
        self.assertTrue(candidate.pop("enable_f4_f3_context_guided_mscb_lite_stage3"))
        reference.pop("enable_residual_fg_mscb_lite_stage3")
        for item in (reference, candidate):
            item.pop("name")
        self.assertEqual(reference, candidate)
        model = build_experiment_model(get_experiment(EXP45), encoder_weights=None)
        self.assertIsInstance(model.residual_fg_mscb_lite_stage3, ResidualFrequencyGuidedMSCBLite)
        result = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(RUNNER), "-DryRun"], cwd=ROOT, text=True, capture_output=True, check=True)
        command = dict(zip(json.loads(result.stdout.strip())["train"][::2], json.loads(result.stdout.strip())["train"][1::2]))
        self.assertEqual(command["--experiment_name"], EXP51)
        self.assertEqual(command["--batch_size"], "32")
        self.assertEqual(command["--enable_f4_f3_context_guided_mscb_lite_stage3"], "True")


if __name__ == "__main__":
    unittest.main()
