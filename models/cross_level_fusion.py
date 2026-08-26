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


class _ContextRecalibration(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden_channels = max(8, channels // reduction)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x * (1.0 + self.channel_attention(x))
        spatial_descriptor = torch.cat(
            (x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)),
            dim=1,
        )
        return x * (1.0 + self.spatial_attention(spatial_descriptor))


class CrossLevelFusionV2(nn.Module):
    """Adjacent-level fusion at each skip's native spatial resolution."""

    context_sources = ((0, 1), (0, 1, 2), (1, 2))

    def __init__(self, channels=(96, 192, 384), fusion_channels=64, reduction=8):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("CrossLevelFusionV2 requires exactly three skip levels")
        attention_channels = max(8, fusion_channels // 4)
        self.input_projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channel_count, fusion_channels, 1, bias=False),
                nn.GroupNorm(1, fusion_channels),
                nn.GELU(),
            )
            for channel_count in channels
        )
        self.context_attention = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(fusion_channels, attention_channels, 1),
                nn.GELU(),
                nn.Conv2d(attention_channels, 1, 1),
            )
            for _ in range(3)
        )
        self.recalibration = nn.ModuleList(
            _ContextRecalibration(fusion_channels, reduction) for _ in range(3)
        )
        self.output_projections = nn.ModuleList(
            nn.Conv2d(fusion_channels, channel_count, 1, bias=False)
            for channel_count in channels
        )
        self.residual_scales = nn.Parameter(torch.full((3,), 1e-3))

    @staticmethod
    def _resize(feature, target_size):
        if feature.shape[-2:] == target_size:
            return feature
        if (feature.shape[-2] >= target_size[0] and
                feature.shape[-1] >= target_size[1]):
            return F.adaptive_avg_pool2d(feature, target_size)
        return F.interpolate(
            feature, size=target_size, mode="bilinear", align_corners=False
        )

    def forward(self, f1, f2, f3):
        originals = (f1, f2, f3)
        projected = tuple(
            projection(feature)
            for projection, feature in zip(self.input_projections, originals)
        )
        outputs = []
        for target_index, source_indices in enumerate(self.context_sources):
            target_size = originals[target_index].shape[-2:]
            aligned = [
                self._resize(projected[source_index], target_size)
                for source_index in source_indices
            ]
            logits = torch.cat(
                [self.context_attention[target_index](feature)
                 for feature in aligned],
                dim=1,
            )
            weights = torch.softmax(logits, dim=1)
            context = sum(
                feature * weights[:, index:index + 1]
                for index, feature in enumerate(aligned)
            )
            context = self.recalibration[target_index](context)
            refinement = self.output_projections[target_index](context)
            outputs.append(
                originals[target_index] +
                self.residual_scales[target_index] * refinement
            )
        return tuple(outputs)
