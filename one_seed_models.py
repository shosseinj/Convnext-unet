"""Model construction for isolated seed-42 decoder/skip experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.architecture_factory import UGBRVariant, UncertaintyRefinementV2Variant
from models.convnext_pretrain import ConvNeXtUNet


class DySample(nn.Module):
    """Pure-PyTorch LP-style DySample upsampler (ICCV 2023 official design)."""

    def __init__(self, in_channels, scale=2, groups=4):
        super().__init__()
        if in_channels < groups or in_channels % groups:
            raise ValueError("in_channels must be divisible by groups")
        self.scale = scale
        self.groups = groups
        self.offset = nn.Conv2d(in_channels, 2 * groups * scale * scale, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.constant_(self.offset.bias, 0)
        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        coordinate = torch.arange(
            (-self.scale + 1) / 2, (self.scale - 1) / 2 + 1
        ) / self.scale
        grid_y, grid_x = torch.meshgrid(coordinate, coordinate, indexing="ij")
        return torch.stack((grid_x, grid_y)).transpose(1, 2).repeat(
            1, self.groups, 1
        ).reshape(1, -1, 1, 1)

    def forward(self, x):
        offset = self.offset(x) * 0.25 + self.init_pos
        batch, _, height, width = offset.shape
        offset = offset.view(batch, 2, -1, height, width)
        coords_h = torch.arange(height, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(width, device=x.device, dtype=x.dtype) + 0.5
        grid_y, grid_x = torch.meshgrid(coords_h, coords_w, indexing="ij")
        coords = torch.stack((grid_x, grid_y)).unsqueeze(1).unsqueeze(0)
        normalizer = torch.tensor(
            [width, height], device=x.device, dtype=x.dtype
        ).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(
            coords.view(batch, -1, height, width), self.scale
        ).view(batch, 2, -1, self.scale * height, self.scale * width)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        sampled = F.grid_sample(
            x.reshape(batch * self.groups, -1, height, width),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        return sampled.view(batch, -1, self.scale * height, self.scale * width)


def main_logits(output):
    if isinstance(output, dict):
        return output["final_logits"]
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def gradient_rms(parameters):
    squared_sum = 0.0
    element_count = 0
    for parameter in parameters:
        if parameter.grad is not None:
            gradient = parameter.grad.detach()
            squared_sum += gradient.square().sum().item()
            element_count += gradient.numel()
    return (squared_sum / element_count) ** 0.5 if element_count else 0.0


def cosine_schedule_epochs(total_epochs, detail_warmup_epochs, decoder_warmup_epochs):
    return max(1, total_epochs - max(detail_warmup_epochs, 0) -
               max(decoder_warmup_epochs, 0))


def build_experiment_model(config, encoder_weights, device=None):
    if config.backbone != "convnext_tiny":
        raise ValueError("One-seed experiments require the ConvNeXt-Tiny encoder")
    model = ConvNeXtUNet(
        weights_path=encoder_weights,
        num_classes=1,
        encoder_depth=[3, 3, 9, 3],
        drop_path_rate=0.25,
        dropout_rate=0.2,
        enable_msc=config.enable_msc,
        skip_mode=config.skip_mode,
        detail_channels=config.detail_channels,
        enable_gdf=config.enable_gdf,
        detail_fusion_mode=config.detail_fusion_mode,
        deep_supervision_heads=config.deep_supervision_heads,
        backbone=config.backbone,
        enable_csaf=config.enable_csaf,
        enable_fafem=config.enable_fafem,
        csaf_version=config.csaf_version,
        fafem_stage1=config.fafem_stage1,
        fafem_stage2=config.fafem_stage2,
        fafem_stage3=config.fafem_stage3,
        enable_gated_skip_stage3=getattr(config, "enable_gated_skip_stage3", False),
        enable_cross_level_fusion=config.enable_cross_level_fusion,
        cross_level_fusion_version=config.cross_level_fusion_version,
        enable_geometry_conv_stage3=getattr(config, "enable_geometry_conv_stage3", False),
        decoder_highres_width=getattr(config, "decoder_highres_width", 96),
        enable_mscb_lite_stage3=getattr(config, "enable_mscb_lite_stage3", False),
        enable_fg_mscb_lite_stage3=getattr(config, "enable_fg_mscb_lite_stage3", False),
        enable_residual_fg_mscb_lite_stage3=getattr(config, "enable_residual_fg_mscb_lite_stage3", False),
        residual_fg_mscb_guidance_init_std=getattr(
            config, "residual_fg_mscb_guidance_init_std", 1e-3
        ),
        residual_fg_mscb_signed_strength=getattr(
            config, "residual_fg_mscb_signed_strength", False
        ),
        residual_fg_mscb_initial_strength=getattr(
            config, "residual_fg_mscb_initial_strength", 0.0
        ),
        enable_deformable_residual_fg_mscb_lite_stage3=getattr(
            config, "enable_deformable_residual_fg_mscb_lite_stage3", False
        ),
        enable_residual_fg_mscb_all_skips=getattr(
            config, "enable_residual_fg_mscb_all_skips", False
        ),
        enable_partial_deformable_residual_fg_mscb_lite_stage3=getattr(
            config, "enable_partial_deformable_residual_fg_mscb_lite_stage3", False
        ),
        enable_f4_f3_context_guided_mscb_lite_stage3=getattr(
            config, "enable_f4_f3_context_guided_mscb_lite_stage3", False
        ),
        enable_mixstyle_stage1_stage2=getattr(
            config, "enable_mixstyle_stage1_stage2", False
        ),
        enable_mscb_lite_stage2=getattr(config, "enable_mscb_lite_stage2", False),
        enable_mscb_lite_stage1=getattr(config, "enable_mscb_lite_stage1", False),
        enable_lka_lite_stage3=getattr(config, "enable_lka_lite_stage3", False),
    )
    if config.upsample_mode == "dysample":
        for decoder in (model.decoder4, model.decoder3, model.decoder2, model.decoder1):
            decoder.upsample = DySample(decoder.conv.depthwise.in_channels)
    elif config.upsample_mode != "bilinear":
        raise ValueError(f"Unsupported upsample mode: {config.upsample_mode}")
    model.experiment_variant = config.to_dict()
    model = UGBRVariant(model) if config.enable_ugbr else model
    if config.uncertainty_refinement_version == "v2":
        model = UncertaintyRefinementV2Variant(model)
    elif config.uncertainty_refinement_version != "none":
        raise ValueError(
            "Unsupported uncertainty refinement version: "
            f"{config.uncertainty_refinement_version}"
        )
    model.experiment_variant = config.to_dict()
    if device is not None:
        model = model.to(device)
    return model
