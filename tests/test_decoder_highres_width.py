import unittest

from ablation_registry import get_experiment


class DecoderHighResolutionWidthExperimentTests(unittest.TestCase):
    def test_decoder_width_ablation_registers_a_wider_high_resolution_path(self):
        config = get_experiment("one_seed_35_fafem_decoder_highres_wide")

        self.assertEqual(config.decoder_highres_width, 120)
        self.assertTrue(config.enable_fafem)
        self.assertFalse(config.enable_gated_skip_stage3)

    def test_fafem_bsei_ablation_uses_bsei_as_its_only_skip_change(self):
        config = get_experiment("one_seed_36_fafem_bsei_warmup_cosine")

        self.assertTrue(config.enable_fafem)
        self.assertEqual(config.skip_mode, "bsei")
        self.assertFalse(config.enable_gated_skip_stage3)
        self.assertEqual(config.decoder_highres_width, 96)


if __name__ == "__main__":
    unittest.main()
