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

    def to_dict(self):
        return asdict(self)


_EXPERIMENTS = {
    values[0]: ExperimentConfig(*values)
    for values in (
        ("01_baseline", False, "normal", 0, False, "none", 0),
        ("02_add_msc", True, "normal", 0, False, "none", 0),
        ("03_add_lrse", False, "bsei", 0, False, "none", 0),
        ("04_add_db", True, "bsei", 32, False, "concatenation", 0),
        ("05_add_gdf", True, "bsei", 32, True, "gdf", 0),
        ("06_full_model", True, "bsei", 32, True, "gdf", 3),
        ("07_full_without_msc", False, "bsei", 32, True, "gdf", 3),
        ("08_full_without_lrse", True, "normal", 32, True, "gdf", 3),
        ("09_gdf_concat", True, "bsei", 32, False, "concatenation", 3),
        ("10_gdf_addition", True, "bsei", 32, False, "addition", 3),
        ("11_full_without_ds", True, "bsei", 32, True, "gdf", 0),
    )
}


def canonical_seeds():
    return 42, 6543, 7777


def get_experiment(name):
    try:
        return _EXPERIMENTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown ablation experiment: {name}") from exc
