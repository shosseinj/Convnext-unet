"""Cross-Scale Attention Fusion for optional ConvNeXt encoder skip refinement."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossScaleAttentionFusion(nn.Module):
    """Fuse every encoder scale into one residual channel/spatial-attended skip."""

    def __init__(self, source_channels, target_index, target_channels):
        super().__init__()
        hidden_channels = max(16, target_channels // 4)
        self.target_index = target_index
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, hidden_channels, 1, bias=False)
            for channels in source_channels
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * len(source_channels), target_channels, 1, bias=False),
            nn.GELU(),
        )
        attention_channels = max(8, target_channels // 8)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(target_channels, attention_channels, 1),
            nn.GELU(),
            nn.Conv2d(attention_channels, target_channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, features):
        target = features[self.target_index]
        target_size = target.shape[-2:]
        projected = []
        for projection, feature in zip(self.projections, features):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature, size=target_size, mode="bilinear", align_corners=False
                )
            projected.append(feature)
        fused = self.fuse(torch.cat(projected, dim=1))
        channel_weights = self.channel_attention(fused)
        spatial_input = torch.cat(
            [fused.mean(dim=1, keepdim=True), fused.amax(dim=1, keepdim=True)],
            dim=1,
        )
        spatial_weights = self.spatial_attention(spatial_input)
        return target + self.gamma * fused * channel_weights * spatial_weights
