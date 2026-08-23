"""Uncertainty-guided boundary refinement for segmentation logits."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class UGBR(nn.Module):
    """Refine initial logits using decoder detail, shallow features and uncertainty."""

    def __init__(self, decoder_channels: int, shallow_channels: int,
                 num_classes: int = 1, hidden_channels: int = 48) -> None:
        super().__init__()
        self.decoder_projection = nn.Sequential(
            nn.Conv2d(decoder_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.shallow_projection = nn.Sequential(
            nn.Conv2d(shallow_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        fused_channels = hidden_channels * 2 + num_classes
        self.boundary_head = nn.Sequential(
            nn.Conv2d(fused_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, num_classes, 1),
        )
        self.refinement_head = nn.Sequential(
            nn.Conv2d(fused_channels + num_classes, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, num_classes, 1),
        )

    @staticmethod
    def uncertainty(initial_logits: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(initial_logits)
        return 1.0 - torch.abs(2.0 * probability - 1.0)

    def forward(self, decoder_feature: torch.Tensor, shallow_encoder_feature: torch.Tensor,
                initial_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        target_size = shallow_encoder_feature.shape[-2:]
        decoder = F.interpolate(self.decoder_projection(decoder_feature), target_size,
                                mode="bilinear", align_corners=False)
        shallow = self.shallow_projection(shallow_encoder_feature)
        initial_low_resolution = F.interpolate(
            initial_logits, target_size, mode="bilinear", align_corners=False
        )
        uncertainty_low_resolution = self.uncertainty(initial_low_resolution)
        fused = torch.cat((decoder, shallow, uncertainty_low_resolution), dim=1)
        boundary_low_resolution = self.boundary_head(fused)
        refinement_low_resolution = self.refinement_head(
            torch.cat((fused, boundary_low_resolution), dim=1)
        )
        full_size = initial_logits.shape[-2:]
        boundary_logits = F.interpolate(
            boundary_low_resolution, full_size, mode="bilinear", align_corners=False
        )
        refinement_logits = F.interpolate(
            refinement_low_resolution, full_size, mode="bilinear", align_corners=False
        )
        uncertainty = self.uncertainty(initial_logits)
        final_logits = initial_logits + refinement_logits
        return {
            "initial_logits": initial_logits,
            "boundary_logits": boundary_logits,
            "refinement_logits": refinement_logits,
            "final_logits": final_logits,
            "uncertainty": uncertainty,
        }
