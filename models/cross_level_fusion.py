"""Lightweight cross-level fusion for ConvNeXt U-Net skip features."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossLevelFusion(nn.Module):
    """Fuse three encoder levels and residually recalibrate every skip."""

    def __init__(self, channels=(96, 192, 384), fusion_channels=96, reduction=8):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("CrossLevelFusion requires exactly three skip levels")
        hidden_channels = max(8, fusion_channels // reduction)
        attention_channels = max(8, fusion_channels // 4)

        self.input_projections = nn.ModuleList(
            nn.Conv2d(channel_count, fusion_channels, 1, bias=False)
            for channel_count in channels
        )
        self.level_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fusion_channels, attention_channels, 1),
            nn.GELU(),
            nn.Conv2d(attention_channels, 1, 1),
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fusion_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, fusion_channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.output_projections = nn.ModuleList(
            nn.Conv2d(fusion_channels, channel_count, 1, bias=False)
            for channel_count in channels
        )
        self.residual_scales = nn.Parameter(torch.zeros(3))

    def forward(self, f1, f2, f3):
        features = (f1, f2, f3)
        target_size = f3.shape[-2:]
        aligned = []
        for feature, projection in zip(features, self.input_projections):
            if feature.shape[-2:] != target_size:
                feature = F.adaptive_avg_pool2d(feature, target_size)
            aligned.append(projection(feature))

        level_logits = torch.cat(
            [self.level_attention(feature) for feature in aligned], dim=1
        )
        level_weights = torch.softmax(level_logits, dim=1)
        fused = sum(
            feature * level_weights[:, index:index + 1]
            for index, feature in enumerate(aligned)
        )

        channel_refined = fused * self.channel_attention(fused)
        spatial_descriptor = torch.cat(
            (
                channel_refined.mean(dim=1, keepdim=True),
                channel_refined.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        refined = channel_refined * self.spatial_attention(spatial_descriptor)

        outputs = []
        for index, (original, projection) in enumerate(
            zip(features, self.output_projections)
        ):
            context = projection(refined)
            if context.shape[-2:] != original.shape[-2:]:
                context = F.interpolate(
                    context,
                    size=original.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            outputs.append(original + self.residual_scales[index] * context)
        return tuple(outputs)
