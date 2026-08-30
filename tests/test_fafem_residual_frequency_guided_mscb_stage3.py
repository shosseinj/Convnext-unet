import unittest

import torch

from ablation_registry import get_experiment
from models.convnext_pretrain import MSCBLite
from one_seed_models import build_experiment_model


EXPERIMENT = "one_seed_44_fafem_residual_frequency_guided_mscb_stage3_warmup_cosine"


class FAFEMResidualFrequencyGuidedMSCBTests(unittest.TestCase):
    def test_residual_frequency_guidance_starts_as_ordinary_mscb(self):
        model = build_experiment_model(get_experiment(EXPERIMENT), encoder_weights=None).eval()
        module = model.residual_fg_mscb_lite_stage3
        self.assertIsNotNone(module)
        self.assertEqual(float(module.guidance_strength.item()), 0.0)
        self.assertIsNone(model.mscb_lite_stage3)
        self.assertIsNone(model.fg_mscb_lite_stage3)

        ordinary = MSCBLite(384).eval()
        ordinary.load_state_dict({
            key: value for key, value in module.state_dict().items()
            if not key.startswith(("guidance_mlp.", "guidance_strength"))
        })
        features = torch.randn(2, 384, 22, 22)
        descriptor = torch.randn(2, 1536)
        with torch.no_grad():
            self.assertTrue(torch.equal(module(features, descriptor), ordinary(features)))

    def test_residual_guidance_has_a_nonzero_learning_gradient_at_initialization(self):
        model = build_experiment_model(get_experiment(EXPERIMENT), encoder_weights=None)
        module = model.residual_fg_mscb_lite_stage3
        features = torch.randn(2, 384, 22, 22, requires_grad=True)
        descriptor = torch.randn(2, 1536)
        module(features, descriptor).mean().backward()
        self.assertIsNotNone(module.guidance_strength.grad)
        self.assertGreater(module.guidance_strength.grad.abs().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
