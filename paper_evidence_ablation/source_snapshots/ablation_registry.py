"""Canonical architecture settings for the PowerShell ablation launchers."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    enable_msc: bool
    skip_mode: str
    detail_channels: int
    enable_gdf: bool
    detail_fusion_mode: str
    deep_supervision_heads: int
    enable_ugbr: bool = False
    upsample_mode: str = "bilinear"
    backbone: str = "convnext_tiny"
    training_precision: str = "fp32"
    max_epochs: int = 150
    encoder_freeze_epochs: int = 10
    enable_csaf: bool = False
    enable_fafem: bool = False
    csaf_version: str = "v1"
    fafem_stage1: bool = False
    fafem_stage2: bool = False
    fafem_stage3: bool = False
    enable_gated_skip_stage3: bool = False
    enable_cross_level_fusion: bool = False
    cross_level_fusion_version: str = "v1"
    unfreeze_schedule: str = "plateau"
    enable_frequency_augmentation: bool = False
    uncertainty_refinement_version: str = "none"
    enable_geometry_conv_stage3: bool = False
    decoder_highres_width: int = 96
    enable_mscb_lite_stage3: bool = False
    enable_lka_lite_stage3: bool = False
    enable_mscb_lite_stage2: bool = False
    enable_mscb_lite_stage1: bool = False
    enable_fg_mscb_lite_stage3: bool = False
    enable_residual_fg_mscb_lite_stage3: bool = False
    residual_fg_mscb_guidance_init_std: float = 1e-3
    residual_fg_mscb_signed_strength: bool = False
    residual_fg_mscb_initial_strength: float = 0.0
    enable_deformable_residual_fg_mscb_lite_stage3: bool = False
    enable_residual_fg_mscb_all_skips: bool = False
    enable_partial_deformable_residual_fg_mscb_lite_stage3: bool = False
    enable_f4_f3_context_guided_mscb_lite_stage3: bool = False
    enable_mixstyle_stage1_stage2: bool = False

    def to_dict(self):
        values = asdict(self)
        if not self.name.startswith("one_seed_"):
            for key in (
                "enable_ugbr", "upsample_mode", "backbone", "training_precision",
                "max_epochs", "encoder_freeze_epochs",
            ):
                values.pop(key)
        if not self.enable_csaf:
            values.pop("enable_csaf")
        if not self.enable_fafem:
            values.pop("enable_fafem")
        if self.csaf_version == "v1":
            values.pop("csaf_version")
        for key in ("fafem_stage1", "fafem_stage2", "fafem_stage3"):
            if not values[key]:
                values.pop(key)
        if not self.enable_gated_skip_stage3:
            values.pop("enable_gated_skip_stage3")
        if not self.enable_cross_level_fusion:
            values.pop("enable_cross_level_fusion")
        if self.cross_level_fusion_version == "v1":
            values.pop("cross_level_fusion_version")
        if self.unfreeze_schedule == "plateau":
            values.pop("unfreeze_schedule")
        if not self.enable_frequency_augmentation:
            values.pop("enable_frequency_augmentation")
        if self.uncertainty_refinement_version == "none":
            values.pop("uncertainty_refinement_version")
        if not self.enable_geometry_conv_stage3:
            values.pop("enable_geometry_conv_stage3")
        if self.decoder_highres_width == 96:
            values.pop("decoder_highres_width")
        if not self.enable_mscb_lite_stage3:
            values.pop("enable_mscb_lite_stage3")
        if not self.enable_fg_mscb_lite_stage3:
            values.pop("enable_fg_mscb_lite_stage3")
        residual_fg_enabled = (
            self.enable_residual_fg_mscb_lite_stage3 or
            self.enable_deformable_residual_fg_mscb_lite_stage3 or
            self.enable_residual_fg_mscb_all_skips or
            self.enable_partial_deformable_residual_fg_mscb_lite_stage3 or
            self.enable_f4_f3_context_guided_mscb_lite_stage3
        )
        if not self.enable_residual_fg_mscb_lite_stage3:
            values.pop("enable_residual_fg_mscb_lite_stage3")
        if not residual_fg_enabled:
            values.pop("residual_fg_mscb_guidance_init_std")
        elif self.residual_fg_mscb_guidance_init_std == 1e-3:
            values.pop("residual_fg_mscb_guidance_init_std")
        if not self.residual_fg_mscb_signed_strength:
            values.pop("residual_fg_mscb_signed_strength")
        if self.residual_fg_mscb_initial_strength == 0.0:
            values.pop("residual_fg_mscb_initial_strength")
        if not self.enable_deformable_residual_fg_mscb_lite_stage3:
            values.pop("enable_deformable_residual_fg_mscb_lite_stage3")
        if not self.enable_residual_fg_mscb_all_skips:
            values.pop("enable_residual_fg_mscb_all_skips")
        if not self.enable_partial_deformable_residual_fg_mscb_lite_stage3:
            values.pop("enable_partial_deformable_residual_fg_mscb_lite_stage3")
        if not self.enable_f4_f3_context_guided_mscb_lite_stage3:
            values.pop("enable_f4_f3_context_guided_mscb_lite_stage3")
        if not self.enable_mixstyle_stage1_stage2:
            values.pop("enable_mixstyle_stage1_stage2")
        if not self.enable_lka_lite_stage3:
            values.pop("enable_lka_lite_stage3")
        if not self.enable_mscb_lite_stage2:
            values.pop("enable_mscb_lite_stage2")
        if not self.enable_mscb_lite_stage1:
            values.pop("enable_mscb_lite_stage1")
        return values


_EXPERIMENTS = {
    values[0]: ExperimentConfig(*values)
    for values in (
        ("01_baseline", False, "normal", 0, False, "none", 0),
        ("02_add_msc", True, "normal", 0, False, "none", 0),
        ("03_add_lrse", False, "bsei", 0, False, "none", 0),
        ("04_add_db", False, "normal", 32, False, "concatenation", 0),
        ("05_add_gdf", False, "normal", 32, True, "gdf", 0),
        ("06_full_model", True, "bsei", 32, True, "gdf", 3),
        ("07_full_without_msc", False, "bsei", 32, True, "gdf", 3),
        ("08_full_without_lrse", True, "normal", 32, True, "gdf", 3),
        ("09_gdf_concat", True, "bsei", 32, False, "concatenation", 3),
        ("10_gdf_addition", True, "bsei", 32, False, "addition", 3),
        ("11_full_without_ds", True, "bsei", 32, True, "gdf", 0),
        ("one_seed_01_baseline", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20),
        ("one_seed_02_baseline_plus_csaf", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, True),
        ("one_seed_03_baseline_plus_fafem", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True),
        ("one_seed_04_baseline_plus_csaf_fafem", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, True, True),
        ("one_seed_05_baseline_plus_fafem_csafv2", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, True, True, "v2"),
        ("one_seed_02_fafem_bottleneck_stage3", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", False, False, True),
        ("one_seed_03_fafem_bottleneck_stage3_stage2", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", False, True, True),
        ("one_seed_04_fafem_bottleneck_stage3_stage2_stage1", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", True, True, True),
        ("one_seed_05_fafem_plus_cross_level_fusion", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", False, False, False, True),
        ("one_seed_06_fafem_plus_cross_level_fusion_v2", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", False, False, False, True, "v2"),
        ("one_seed_07_fafem_plus_ugbr", False, "normal", 0, False, "none", 0, True, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True),
        ("one_seed_08_fafem_plus_dysample", False, "normal", 0, False, "none", 0, False, "dysample", "convnext_tiny", "amp_fp16", 200, 20, False, True),
        ("one_seed_09_fafem_plus_gated_skips", False, "attention_gate", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True),
        ("one_seed_10_fafem_plus_detail_branch", False, "normal", 32, False, "concatenation", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True),
        ("one_seed_11_fafem_plus_msc", True, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True),
        ("one_seed_12_fafem_fixed_unfreeze", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", False, False, False, False, "v1", "fixed"),
        ("one_seed_13_fafem_ugbr_fixed_unfreeze", False, "normal", 0, False, "none", 0, True, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", False, False, False, False, "v1", "fixed"),
        ("one_seed_14_fafem_dysample_fixed_unfreeze", False, "normal", 0, False, "none", 0, False, "dysample", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", False, False, False, False, "v1", "fixed"),
        ("one_seed_15_fafem_gated_skips_fixed_unfreeze", False, "attention_gate", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", False, False, False, False, "v1", "fixed"),
        ("one_seed_16_fafem_detail_branch_fixed_unfreeze", False, "normal", 32, False, "concatenation", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", False, False, False, False, "v1", "fixed"),
        ("one_seed_17_fafem_msc_fixed_unfreeze", True, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", False, False, False, False, "v1", "fixed"),
        ("one_seed_18_fafem_dysample_detail_fixed_unfreeze", False, "normal", 32, False, "concatenation", 0, False, "dysample", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", False, False, False, False, "v1", "fixed"),
        ("one_seed_19_fafem_clfv2_fixed_unfreeze", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", False, False, False, True, "v2", "fixed"),
        ("one_seed_20_fafem_clfv1_fixed_unfreeze", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", False, False, False, True, "v1", "fixed"),
        ("one_seed_21_fafem_dysample_detail_clfv2_fixed_unfreeze", False, "normal", 32, False, "concatenation", 0, False, "dysample", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", False, False, False, True, "v2", "fixed"),
        ("one_seed_22_fafem_dysample_detail_clfv1_fixed_unfreeze", False, "normal", 32, False, "concatenation", 0, False, "dysample", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", False, False, False, True, "v1", "fixed"),
        ("one_seed_23_fafem_detail_clfv2_fixed_unfreeze", False, "normal", 32, False, "concatenation", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", False, False, False, True, "v2", "fixed"),
        ("one_seed_24_fafem_detail_clfv2_optimized_training", False, "normal", 32, False, "concatenation", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", False, False, False, True, "v2", "fixed"),
        ("one_seed_25_fafem_detail_clfv2_warmup_cosine", False, "normal", 32, False, "concatenation", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", False, False, False, True, "v2", "fixed"),
        ("one_seed_26_fafem_detail_clfv2_layerwise_cosine", False, "normal", 32, False, "concatenation", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 160, 0, False, True, "v1", False, False, False, True, "v2", "none"),
        ("one_seed_27_fafem_detail_clfv2_deep_supervision_layerwise_cosine", False, "normal", 32, False, "concatenation", 2, False, "bilinear", "convnext_tiny", "amp_fp16", 220, 0, False, True, "v1", False, False, False, True, "v2", "none"),
        ("one_seed_28_fafem_detail_clfv2_ds_anneal_layerwise_cosine", False, "normal", 32, False, "concatenation", 2, False, "bilinear", "convnext_tiny", "amp_fp16", 160, 0, False, True, "v1", False, False, False, True, "v2", "none"),
        ("one_seed_29_fafem_detail_clfv2_ds_anneal_weighted", False, "normal", 32, False, "concatenation", 2, False, "bilinear", "convnext_tiny", "amp_fp16", 160, 0, False, True, "v1", False, False, False, True, "v2", "none"),
        ("one_seed_30_fafem_fal_uncertainty_refinement", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 180, 5, False, True, "v1", False, False, False, False, "v1", "none", True, "v2"),
        ("one_seed_31_fafem_mild_fal", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20, False, True, "v1", False, False, False, False, "v1", "plateau", True, "none"),
        ("one_seed_fafem_plus_geometry_conv_stage3", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 120, 0, False, True, "v1", False, False, False, False, "v1", "none", False, "none", True),
        ("one_seed_32_fafem_layerwise_warmup_cosine", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 160, 0, False, True, "v1", False, False, False, False, "v1", "none", False, "none"),
        ("one_seed_33_fafem_warmup_cosine", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, False, False, "v1", "none", False, "none", False),
        ("one_seed_34_fafem_gated_skip_stage3", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, True, False, "v1", "none", False, "none", False),
        ("one_seed_35_fafem_decoder_highres_wide", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, False, False, "v1", "none", False, "none", False, 120),
        ("one_seed_36_fafem_bsei_warmup_cosine", False, "bsei", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, False, False, "v1", "none", False, "none", False, 96),
        ("one_seed_37_fafem_mscb_lite_stage3_warmup_cosine", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, False, False, "v1", "none", False, "none", False, 96, True),
        ("one_seed_38_fafem_mscb_lite_stage3_cosine_refinement", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, False, False, "v1", "none", False, "none", False, 96, True),
        ("one_seed_39_fafem_mscb_lite_detail_warmup_cosine", False, "normal", 32, False, "concatenation", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, False, False, "v1", "none", False, "none", False, 96, True),
        ("one_seed_40_fafem_lka_lite_stage3_mscb_warmup_cosine", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, False, False, "v1", "none", False, "none", False, 96, True, True),
        ("one_seed_41_fafem_mscb_lite_stage3_stage2_warmup_cosine", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, False, False, "v1", "none", False, "none", False, 96, True, False, True),
        ("one_seed_42_fafem_mscb_lite_stage3_stage2_stage1_warmup_cosine", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, False, False, "v1", "none", False, "none", False, 96, True, False, True, True),
        ("one_seed_43_fafem_frequency_guided_mscb_stage3_warmup_cosine", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, False, False, "v1", "none", False, "none", False, 96, False, False, False, False, True),
        ("one_seed_44_fafem_residual_frequency_guided_mscb_stage3_warmup_cosine", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, False, False, "v1", "none", False, "none", False, 96, False, False, False, False, False, True),
        ("one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, False, False, "v1", "none", False, "none", False, 96, False, False, False, False, False, True, 0.0, False, 5e-2),
        ("one_seed_46_fafem_residual_frequency_guided_deformable_mscb_stage3_warmup_cosine", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, False, False, "v1", "none", False, "none", False, 96, False, False, False, False, False, False, 0.0, False, 5e-2, True),
        ("one_seed_47_fafem_residual_frequency_guided_deformable_mscb_stage3_full_cosine", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, False, False, "v1", "none", False, "none", False, 96, False, False, False, False, False, False, 0.0, False, 5e-2, True),
        ("one_seed_48_fafem_residual_frequency_guided_mscb_all_skips_warmup_cosine", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, False, False, "v1", "none", False, "none", False, 96, False, False, False, False, False, False, 0.0, False, 5e-2, False, True),
        ("one_seed_49_fafem_residual_frequency_guided_mscb_all_skips_batch32", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, False, False, "v1", "none", False, "none", False, 96, False, False, False, False, False, False, 0.0, False, 5e-2, False, True),
        ("one_seed_50_fafem_residual_frequency_guided_partial_deformable_mscb_stage3_warmup_cosine", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, False, False, "v1", "none", False, "none", False, 96, False, False, False, False, False, False, 0.0, False, 5e-2, False, False, True),
        ("one_seed_51_fafem_f4_f3_context_guided_mscb_stage3_batch32", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, False, False, "v1", "none", False, "none", False, 96, False, False, False, False, False, False, 0.0, False, 5e-2, False, False, False, True),
        ("one_seed_52_fafem_residual_frequency_guided_mscb_stage3_mixstyle12_warmup_cosine", False, "normal", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 0, False, True, "v1", False, False, False, False, False, "v1", "none", False, "none", False, 96, False, False, False, False, False, True, 0.0, False, 5e-2, False, False, False, False, True),
        ("one_seed_02_add_ugbr", False, "normal", 0, False, "none", 0, True, "bilinear", "convnext_tiny", "amp_fp16", 200, 20),
        ("one_seed_03_gated_skips", False, "attention_gate", 0, False, "none", 0, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20),
        ("one_seed_04_deep_supervision", False, "normal", 0, False, "none", 2, False, "bilinear", "convnext_tiny", "amp_fp16", 200, 20),
        ("one_seed_05_dysample", False, "normal", 0, False, "none", 0, False, "dysample", "convnext_tiny", "amp_fp16", 200, 20),
    )
}


def canonical_seeds():
    return 42, 6543, 7777


def get_experiment(name):
    try:
        return _EXPERIMENTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown ablation experiment: {name}") from exc
