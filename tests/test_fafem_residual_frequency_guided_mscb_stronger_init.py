import unittest

import torch

from ablation_registry import get_experiment
from models.convnext_pretrain import MSCBLite
from one_seed_models import build_experiment_model


EXPERIMENT = "one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine"


class StrongerResidualFGMSCBTests(unittest.TestCase):
    def test_identity_output_and_nontrivial_initial_guidance_seed(self):
        model = build_experiment_model(get_experiment(EXPERIMENT), encoder_weights=None).eval()
        module = model.residual_fg_mscb_lite_stage3
        self.assertAlmostEqual(float(module.guidance_strength.item()), 0.05, places=6)
        self.assertEqual(module.guidance_init_std, 0.0)
        self.assertFalse(module.signed_strength)

        ordinary = MSCBLite(384).eval()
        ordinary.load_state_dict({
            key: value for key, value in module.state_dict().items()
            if not key.startswith(("guidance_mlp.", "guidance_strength"))
        })
        features = torch.randn(2, 384, 22, 22)
        descriptor = torch.randn(2, 1536)
        with torch.no_grad():
            self.assertTrue(torch.equal(module(features, descriptor), ordinary(features)))
            alpha = module.branch_weights(descriptor)
        self.assertTrue(torch.allclose(alpha.sum(dim=1), torch.full((2,), 3.0)))
        self.assertTrue(torch.equal(alpha, torch.ones_like(alpha)))

    def test_guidance_mlp_gets_usable_initial_gradient(self):
        model = build_experiment_model(get_experiment(EXPERIMENT), encoder_weights=None)
        module = model.residual_fg_mscb_lite_stage3
        output = module(torch.randn(2, 384, 22, 22), torch.randn(2, 1536))
        output.mean().backward()
        self.assertGreater(module.guidance_mlp[-1].weight.grad.abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
