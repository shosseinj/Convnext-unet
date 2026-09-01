"""Lightweight Frequency-Aware Feature Enhancement Module."""

import torch
import torch.nn as nn


class FrequencyAwareFeatureEnhancement(nn.Module):
    """Enhance pooled low frequencies and residual high frequencies."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_channels = max(8, channels // reduction)
        self.smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.low_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GELU(),
        )
        self.high_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GELU(),
        )
        self.branch_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels * 2, 1),
            nn.Sigmoid(),
        )
        self.refine = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward_with_frequency_descriptor(self, x):
        """Return the FAFEM output and its pooled low/high frequency descriptor."""
        low = self.smooth(x)
        high = x - low
        low = self.low_enhance(low)
        high = self.high_enhance(high)
        descriptor = torch.flatten(
            torch.nn.functional.adaptive_avg_pool2d(
                torch.cat([low, high], dim=1), output_size=1
            ),
            start_dim=1,
        )
        low_weight, high_weight = self.branch_weight(
            torch.cat([low, high], dim=1)
        ).chunk(2, dim=1)
        fused = self.refine(low * low_weight + high * high_weight)
        return x + self.gamma * fused, descriptor

    def forward(self, x):
        output, _ = self.forward_with_frequency_descriptor(x)
        return output
