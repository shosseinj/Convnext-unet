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
