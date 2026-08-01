"""Config-driven ConvNeXt-UNet architecture variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable

import yaml

from .convnext_pretrain import ConvNeXtUNet


@dataclass(frozen=True)
class VariantConfig:
    id: str
    msc: bool
    skip: str
    detail_channels: int
    gdf: bool
    deep_supervision_heads: int
    msc_dilations: tuple[int, ...] = (1, 3, 5)
    gdf_mode: str = "gdf"
    backbone: str = "convnext_tiny"

    def validate(self) -> None:
        if self.skip not in {"normal", "attention_gate", "bsei"}:
            raise ValueError(f"Variant {self.id}: unsupported skip mode {self.skip}")
        if self.detail_channels < 0:
            raise ValueError(f"Variant {self.id}: detail_channels cannot be negative")
        if self.gdf and not self.detail_channels:
            raise ValueError(f"Variant {self.id}: GDF requires a detail branch")
        if self.gdf_mode not in {"addition", "concatenation", "attention_fusion", "gdf"}:
            raise ValueError(f"Variant {self.id}: unsupported detail fusion mode {self.gdf_mode}")
        if self.backbone not in {"resnet34", "efficientnet_b0", "convnext_tiny"}:
            raise ValueError(f"Variant {self.id}: unsupported backbone {self.backbone}")
        if self.deep_supervision_heads not in {0, 1, 2, 3}:
            raise ValueError(f"Variant {self.id}: deep supervision heads must be 0..3")
        if not self.msc_dilations or len(self.msc_dilations) > 5:
            raise ValueError(f"Variant {self.id}: MSC requires 1..5 dilation branches")
        if any(not isinstance(value, int) or value <= 0 for value in self.msc_dilations):
            raise ValueError(f"Variant {self.id}: MSC dilations must be positive integers")


def load_variant_configs(path: Path | str = "configs/ablation_matrix.yaml",
                         sections: tuple[str, ...] = ("incremental",)) -> Dict[str, VariantConfig]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    variants: Dict[str, VariantConfig] = {}
    for section in sections:
        for raw in document.get(section, ()):
            raw = dict(raw)
            if "msc_dilations" in raw:
                raw["msc_dilations"] = tuple(raw["msc_dilations"])
            config = VariantConfig(**raw)
            config.validate()
            if config.id in variants:
                raise ValueError(f"Duplicate variant id: {config.id}")
            variants[config.id] = config
    return variants


def build_variant(config: VariantConfig, weights_path=None, num_classes=1,
                  encoder_depth: Iterable[int] = (3, 3, 9, 3),
                  drop_path_rate=0.1, dropout_rate=0.1) -> ConvNeXtUNet:
    config.validate()
    model = ConvNeXtUNet(
        weights_path=weights_path,
        num_classes=num_classes,
        encoder_depth=list(encoder_depth),
        drop_path_rate=drop_path_rate,
        dropout_rate=dropout_rate,
        enable_msc=config.msc,
        skip_mode=config.skip,
        detail_channels=config.detail_channels,
        enable_gdf=config.gdf,
        deep_supervision_heads=config.deep_supervision_heads,
        msc_dilations=config.msc_dilations,
        detail_fusion_mode=config.gdf_mode if config.gdf else None,
        backbone=config.backbone,
    )
    model.experiment_variant = asdict(config)
    return model


def build_variant_by_id(variant_id: str, matrix_path="configs/ablation_matrix.yaml", **kwargs):
    variants = load_variant_configs(matrix_path, sections=("incremental", "controls"))
    if variant_id not in variants:
        raise KeyError(f"Unknown variant '{variant_id}'. Available: {', '.join(variants)}")
    return build_variant(variants[variant_id], **kwargs)
