"""Identity-safe uncertainty and boundary refinement for segmentation logits."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyBoundaryRefinementV2(nn.Module):
    def __init__(self, decoder_channels=96, shallow_channels=96, hidden_channels=32):
        super().__init__()
        self.decoder_projection = nn.Sequential(
            nn.Conv2d(decoder_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels), nn.GELU(),
        )
        self.shallow_projection = nn.Sequential(
            nn.Conv2d(shallow_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels), nn.GELU(),
        )
        fused_channels = hidden_channels * 2 + 1
        self.boundary_head = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, 3, padding=1,
                      groups=fused_channels, bias=False),
            nn.Conv2d(fused_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels), nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        refinement_channels = fused_channels + 1
        self.refinement_features = nn.Sequential(
            nn.Conv2d(refinement_channels, refinement_channels, 3, padding=1,
                      groups=refinement_channels, bias=False),
            nn.Conv2d(refinement_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels), nn.GELU(),
        )
        self.refinement_output = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.zeros_(self.refinement_output.weight)
        nn.init.zeros_(self.refinement_output.bias)

    @staticmethod
    def uncertainty(logits):
        probability = torch.sigmoid(logits)
        return 1.0 - torch.abs(2.0 * probability - 1.0)

    def forward(self, decoder_feature, shallow_feature, initial_logits):
        target_size = shallow_feature.shape[-2:]
        decoder = F.interpolate(self.decoder_projection(decoder_feature), target_size,
                                mode="bilinear", align_corners=False)
        shallow = self.shallow_projection(shallow_feature)
        initial_low = F.interpolate(initial_logits, target_size, mode="bilinear",
                                    align_corners=False)
        fused = torch.cat((decoder, shallow, self.uncertainty(initial_low)), dim=1)
        boundary_low = self.boundary_head(fused)
        refinement_low = self.refinement_output(
            self.refinement_features(torch.cat((fused, boundary_low), dim=1))
        )
        full_size = initial_logits.shape[-2:]
        boundary_logits = F.interpolate(boundary_low, full_size, mode="bilinear",
                                        align_corners=False)
        refinement_logits = F.interpolate(refinement_low, full_size, mode="bilinear",
                                          align_corners=False)
        return {
            "initial_logits": initial_logits,
            "boundary_logits": boundary_logits,
            "refinement_logits": refinement_logits,
            "final_logits": initial_logits + refinement_logits,
            "uncertainty": self.uncertainty(initial_logits),
        }
