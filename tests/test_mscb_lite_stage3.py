import unittest

import torch

from ablation_registry import get_experiment
import models.convnext_pretrain as convnext_pretrain


class MSCBLiteStage3ContractTests(unittest.TestCase):
    def test_registered_ablation_is_fafem_only_plus_stage3_mscb_lite(self):
        config = get_experiment("one_seed_37_fafem_mscb_lite_stage3_warmup_cosine")

        self.assertTrue(config.enable_fafem)
        self.assertTrue(config.enable_mscb_lite_stage3)
        self.assertEqual(config.skip_mode, "normal")
        self.assertFalse(config.enable_msc)
        self.assertFalse(config.enable_ugbr)
        self.assertFalse(config.enable_gated_skip_stage3)
        self.assertFalse(config.enable_geometry_conv_stage3)

    def test_mscb_lite_preserves_stage3_feature_shape(self):
        module_class = getattr(convnext_pretrain, "MSCBLite", None)
        self.assertIsNotNone(module_class)
        module = module_class(384).eval()
        with torch.no_grad():
            output = module(torch.randn(1, 384, 22, 22))

        self.assertEqual(tuple(output.shape), (1, 384, 22, 22))
        self.assertIsInstance(module.pconv1[1], torch.nn.BatchNorm2d)
        self.assertIsInstance(module.pconv1[2], torch.nn.ReLU6)
        self.assertEqual(
            [branch[0].kernel_size for branch in module.dwconvs],
            [(1, 1), (3, 3), (5, 5)],
        )


if __name__ == "__main__":
    unittest.main()
