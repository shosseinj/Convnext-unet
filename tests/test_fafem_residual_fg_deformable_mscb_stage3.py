import unittest

import torch

from ablation_registry import get_experiment
from models.convnext_pretrain import (
    DeformableDepthwiseMSCBBranch,
    ResidualFrequencyGuidedMSCBLite,
)
from one_seed_models import build_experiment_model


EXP45 = "one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine"
EXP46 = "one_seed_46_fafem_residual_frequency_guided_deformable_mscb_stage3_warmup_cosine"


class DeformableResidualFGMSCBTests(unittest.TestCase):
    def test_registry_delta_is_only_deformable_branch_replacement(self):
        reference = get_experiment(EXP45).to_dict()
        candidate = get_experiment(EXP46).to_dict()
        self.assertTrue(candidate.pop("enable_deformable_residual_fg_mscb_lite_stage3"))
        reference.pop("enable_residual_fg_mscb_lite_stage3")
        for config in (reference, candidate):
            config.pop("name")
        self.assertEqual(reference, candidate)

    def test_zero_offset_branch_matches_ordinary_depthwise_branch(self):
        ordinary = ResidualFrequencyGuidedMSCBLite(384, 1536).dwconvs[2].eval()
        deformable = DeformableDepthwiseMSCBBranch(768, 5).eval()
        deformable.weight.data.copy_(ordinary[0].weight.data)
        deformable.norm.load_state_dict(ordinary[1].state_dict())
        data = torch.randn(2, 768, 22, 22)
        with torch.no_grad():
            self.assertTrue(torch.allclose(deformable(data), ordinary(data), atol=2e-5, rtol=2e-4))

    def test_exp46_has_three_deformable_branches_and_preserves_shape(self):
        model = build_experiment_model(get_experiment(EXP46), encoder_weights=None)
        module = model.deformable_residual_fg_mscb_lite_stage3
        self.assertIsNotNone(module)
        self.assertIsNone(model.residual_fg_mscb_lite_stage3)
        self.assertEqual(len(module.dwconvs), 3)
        self.assertTrue(all(isinstance(branch, DeformableDepthwiseMSCBBranch) for branch in module.dwconvs))
        for branch in module.dwconvs:
            self.assertTrue(torch.equal(branch.offset.weight, torch.zeros_like(branch.offset.weight)))
            self.assertTrue(torch.equal(branch.offset.bias, torch.zeros_like(branch.offset.bias)))
        features = torch.randn(1, 384, 22, 22)
        descriptor = torch.randn(1, 1536)
        output = module(features, descriptor)
        self.assertEqual(output.shape, features.shape)
        alpha = module.branch_weights(descriptor)
        self.assertTrue(torch.allclose(alpha.sum(dim=1), torch.full((1,), 3.0)))

    def test_offset_and_guidance_strength_receive_gradients(self):
        model = build_experiment_model(get_experiment(EXP46), encoder_weights=None)
        module = model.deformable_residual_fg_mscb_lite_stage3
        output = module(torch.randn(2, 384, 22, 22), torch.randn(2, 1536))
        output.square().mean().backward()
        self.assertTrue(all(branch.offset.weight.grad is not None for branch in module.dwconvs))
        self.assertGreater(sum(branch.offset.weight.grad.abs().sum().item() for branch in module.dwconvs), 0.0)
        self.assertIsNotNone(module.guidance_strength.grad)


if __name__ == "__main__":
    unittest.main()
