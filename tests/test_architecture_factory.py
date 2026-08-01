import unittest

from models.architecture_factory import VariantConfig, load_variant_configs


class ArchitectureFactoryTests(unittest.TestCase):
    def test_incremental_matrix_is_complete(self):
        variants = load_variant_configs()
        self.assertEqual(list(variants), [
            "baseline", "baseline_msc", "baseline_msc_bsei",
            "baseline_msc_bsei_detail", "baseline_msc_bsei_detail_gdf", "full",
        ])

    def test_bsei_variant_uses_bsei_skip(self):
        variants = load_variant_configs()
        self.assertEqual(variants["baseline_msc_bsei"].skip, "bsei")

    def test_gdf_requires_detail(self):
        config = VariantConfig("invalid", True, "bsei", 0, True, 0)
        with self.assertRaisesRegex(ValueError, "requires a detail branch"):
            config.validate()

    def test_deep_supervision_range(self):
        config = VariantConfig("invalid", True, "bsei", 32, True, 4)
        with self.assertRaisesRegex(ValueError, "must be 0..3"):
            config.validate()

    def test_msc_branch_and_dilation_controls(self):
        for dilations in ((1, 3), (1, 3, 5), (1, 3, 5, 7), (1, 3, 5, 7, 9),
                          (1, 2, 3), (1, 3, 7)):
            config = VariantConfig("control", True, "normal", 0, False, 0, dilations)
            config.validate()
        with self.assertRaisesRegex(ValueError, "positive integers"):
            VariantConfig("invalid", True, "normal", 0, False, 0, (0, 3)).validate()

    def test_implemented_control_matrix_is_loadable(self):
        controls = load_variant_configs(sections=("controls",))
        self.assertEqual(len(controls), 14)
        self.assertEqual(controls["control_msc_branches_5"].msc_dilations, (1, 3, 5, 7, 9))
        self.assertEqual(controls["control_detail_16"].detail_channels, 16)
        self.assertEqual(controls["control_ds_2"].deep_supervision_heads, 2)
        self.assertEqual(controls["control_skip_attention_gate"].skip, "attention_gate")
        self.assertEqual(controls["control_gdf_addition"].gdf_mode, "addition")
        self.assertEqual(controls["control_backbone_resnet34"].backbone, "resnet34")


if __name__ == "__main__":
    unittest.main()
