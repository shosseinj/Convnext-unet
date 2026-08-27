"""Config-driven ConvNeXt-UNet architecture variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable

import torch.nn as nn
import yaml

from .convnext_pretrain import ConvNeXtUNet
from .ugbr import UGBR
from .uncertainty_refinement_v2 import UncertaintyBoundaryRefinementV2


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
    ugbr: bool = False

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


class UGBRVariant(nn.Module):
    """Optional adapter that preserves the untouched base model implementation."""

    def __init__(self, base_model: ConvNeXtUNet, num_classes: int = 1) -> None:
        super().__init__()
        self.base_model = base_model
        self.ugbr = UGBR(96, 96, num_classes=num_classes)
        self._decoder_feature = None
        self._shallow_feature = None
        self.base_model.final_refine.register_forward_pre_hook(self._capture_decoder)
        self.base_model.encoder.register_forward_hook(self._capture_encoder)
        self.experiment_variant = base_model.experiment_variant

    @property
    def encoder(self):
        """Expose the canonical encoder without registering a second module alias."""
        return self.base_model.encoder

    @property
    def variant_config(self):
        return self.base_model.variant_config

    @property
    def fafem(self):
        return self.base_model.fafem

    def named_parameters(self, prefix: str = "", recurse: bool = True,
                         remove_duplicate: bool = True):
        """Present wrapped base-model names through the existing public contract."""
        for name, parameter in super().named_parameters(
                prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate):
            base_prefix = f"{prefix}.base_model." if prefix else "base_model."
            public_prefix = f"{prefix}." if prefix else ""
            if name.startswith(base_prefix):
                name = public_prefix + name[len(base_prefix):]
            yield name, parameter

    def _capture_decoder(self, _module, inputs) -> None:
        self._decoder_feature = inputs[0]

    def _capture_encoder(self, _module, _inputs, output) -> None:
        self._shallow_feature = output[0]

    def forward(self, x):
        base_output = self.base_model(x)
        initial_logits = base_output[0] if isinstance(base_output, tuple) else base_output
        if self._decoder_feature is None or self._shallow_feature is None:
            raise RuntimeError("UGBR feature hooks did not capture required tensors")
        return self.ugbr(self._decoder_feature, self._shallow_feature, initial_logits)

    def freeze_encoder(self):
        return self.base_model.freeze_encoder()

    def unfreeze_encoder(self):
        return self.base_model.unfreeze_encoder()

    def get_encoder_params(self):
        return self.base_model.get_encoder_params()

    def get_decoder_params(self):
        return (parameter for name, parameter in self.named_parameters()
                if not name.startswith("encoder."))


class UncertaintyRefinementV2Variant(nn.Module):
    """Identity-safe refinement adapter without changing the base U-Net."""

    def __init__(self, base_model: ConvNeXtUNet) -> None:
        super().__init__()
        self.base_model = base_model
        self.uncertainty_refinement = UncertaintyBoundaryRefinementV2()
        self._decoder_feature = None
        self._shallow_feature = None
        self.base_model.final_refine.register_forward_pre_hook(self._capture_decoder)
        self.base_model.encoder.register_forward_hook(self._capture_encoder)
        self.experiment_variant = base_model.experiment_variant

    @property
    def encoder(self):
        return self.base_model.encoder

    @property
    def variant_config(self):
        return self.base_model.variant_config

    @property
    def fafem(self):
        return self.base_model.fafem

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        for name, parameter in super().named_parameters(
                prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate):
            base_prefix = f"{prefix}.base_model." if prefix else "base_model."
            public_prefix = f"{prefix}." if prefix else ""
            if name.startswith(base_prefix):
                name = public_prefix + name[len(base_prefix):]
            yield name, parameter

    def _capture_decoder(self, _module, inputs):
        self._decoder_feature = inputs[0]

    def _capture_encoder(self, _module, _inputs, output):
        self._shallow_feature = output[0]

    def forward(self, x):
        initial_logits = self.base_model(x)
        if isinstance(initial_logits, (tuple, list)):
            initial_logits = initial_logits[0]
        if self._decoder_feature is None or self._shallow_feature is None:
            raise RuntimeError("Refinement hooks did not capture required tensors")
        return self.uncertainty_refinement(
            self._decoder_feature, self._shallow_feature, initial_logits
        )

    def freeze_encoder(self):
        return self.base_model.freeze_encoder()

    def unfreeze_encoder(self):
        return self.base_model.unfreeze_encoder()

    def get_encoder_params(self):
        return self.base_model.get_encoder_params()

    def get_decoder_params(self):
        return (parameter for name, parameter in self.named_parameters()
                if not name.startswith("encoder."))


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
    return UGBRVariant(model, num_classes=num_classes) if config.ugbr else model


def build_variant_by_id(variant_id: str, matrix_path="configs/ablation_matrix.yaml", **kwargs):
    variants = load_variant_configs(matrix_path, sections=("incremental", "controls", "pilot"))
    if variant_id not in variants:
        raise KeyError(f"Unknown variant '{variant_id}'. Available: {', '.join(variants)}")
    return build_variant(variants[variant_id], **kwargs)
