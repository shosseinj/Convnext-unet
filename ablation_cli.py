"""Lightweight CLI contract shared by main_torch and ablation launchers."""


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got: {value}")


def add_ablation_arguments(parser):
    parser.add_argument("--experiment_name")
    parser.add_argument("--seed_dir")
    parser.add_argument("--best_checkpoint_path")
    parser.add_argument("--training_history_path")
    parser.add_argument("--training_summary_path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder_weights", default="./convnext_tiny_22k_1k_384.pth")
    parser.add_argument("--enable_msc", type=parse_bool, default=True)
    parser.add_argument(
        "--unfreeze_schedule", choices=("plateau", "fixed", "none"), default="plateau"
    )
    parser.add_argument("--enable_ugbr", type=parse_bool, default=False)
    parser.add_argument(
        "--upsample_mode", choices=("bilinear", "dysample"), default="bilinear"
    )
    parser.add_argument("--skip_mode", choices=("normal", "attention_gate", "bsei"), default="normal")
    parser.add_argument("--detail_channels", type=int, choices=(0, 16, 32, 64), default=0)
    parser.add_argument("--enable_gdf", type=parse_bool, default=False)
    parser.add_argument("--enable_csaf", type=parse_bool, default=False)
    parser.add_argument("--enable_fafem", type=parse_bool, default=False)
    parser.add_argument("--fafem_stage1", type=parse_bool, default=False)
    parser.add_argument("--fafem_stage2", type=parse_bool, default=False)
    parser.add_argument("--fafem_stage3", type=parse_bool, default=False)
    parser.add_argument("--enable_gated_skip_stage3", type=parse_bool, default=False)
    parser.add_argument("--enable_cross_level_fusion", type=parse_bool, default=False)
    parser.add_argument(
        "--cross_level_fusion_version", choices=("v1", "v2"), default="v1"
    )
    parser.add_argument("--enable_geometry_conv_stage3", type=parse_bool, default=False)
    parser.add_argument("--decoder_highres_width", type=int, choices=(96, 120), default=96)
    parser.add_argument("--enable_mscb_lite_stage3", type=parse_bool, default=False)
    parser.add_argument("--enable_mscb_lite_stage2", type=parse_bool, default=False)
    parser.add_argument("--enable_mscb_lite_stage1", type=parse_bool, default=False)
    parser.add_argument("--enable_lka_lite_stage3", type=parse_bool, default=False)
    parser.add_argument("--csaf_version", choices=("v1", "v2"), default="v1")
    parser.add_argument(
        "--detail_fusion_mode",
        choices=("none", "addition", "concatenation", "attention_fusion", "gdf"),
        default="none",
    )
    parser.add_argument("--deep_supervision_heads", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--deep_supervision_schedule", choices=("constant", "anneal"), default="constant")
    parser.add_argument("--deep_supervision_anneal_start", type=int, default=96)
    parser.add_argument("--deep_supervision_anneal_end", type=int, default=128)
    parser.add_argument("--sampling_mode", choices=("uniform", "lesion_size_weighted"), default="uniform")
    parser.add_argument("--enable_frequency_augmentation", type=parse_bool, default=False)
    parser.add_argument("--frequency_max_probability", type=float, default=0.5)
    parser.add_argument("--frequency_max_mix", type=float, default=0.5)
    parser.add_argument("--frequency_region_min", type=float, default=0.01)
    parser.add_argument("--frequency_region_max", type=float, default=0.05)
    parser.add_argument("--frequency_constant_fraction", type=float, default=0.70)
    parser.add_argument("--frequency_anneal_end_fraction", type=float, default=0.80)
    parser.add_argument("--uncertainty_refinement_version", choices=("none", "v2"), default="none")
    parser.add_argument("--auto_resume", type=parse_bool, default=False)
    parser.add_argument("--amp", type=parse_bool, default=False)
    return parser
