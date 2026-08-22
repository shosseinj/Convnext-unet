import unittest

from ablation_registry import canonical_seeds, get_experiment


class AblationRegistryTests(unittest.TestCase):
    def test_canonical_seeds_are_requested_set(self):
        self.assertEqual(canonical_seeds(), (42, 6543, 7777))

    def test_full_model_configuration(self):
        cfg = get_experiment("06_full_model")
        self.assertEqual(
            (cfg.enable_msc, cfg.skip_mode, cfg.detail_channels,
             cfg.enable_gdf, cfg.detail_fusion_mode, cfg.deep_supervision_heads),
            (True, "bsei", 32, True, "gdf", 3),
        )

    def test_unknown_experiment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown ablation experiment"):
            get_experiment("missing")


if __name__ == "__main__":
    unittest.main()
