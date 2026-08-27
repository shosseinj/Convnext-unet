from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryDeformableConv(nn.Module):
    """Lightweight deformable refinement block for a single encoder stage."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.offset = nn.Conv2d(channels, 2 * kernel_size * kernel_size, 3, padding=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset(x)
        batch, _, height, width = offset.shape
        offset = offset.view(batch, self.kernel_size * self.kernel_size, 2, height, width)

        coords_h = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
        coords_w = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(coords_h, coords_w, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), dim=-1)
        base_grid = base_grid.unsqueeze(0).unsqueeze(1).expand(batch, self.kernel_size * self.kernel_size, height, width, 2)

        offset = torch.tanh(offset.permute(0, 1, 3, 4, 2)) * (2.0 / max(height, width))
        sample_grid = base_grid + offset

        sampled = []
        for index in range(self.kernel_size * self.kernel_size):
            sampled.append(
                F.grid_sample(
                    x,
                    sample_grid[:, index],
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
            )
        sampled = torch.stack(sampled, dim=2).mean(dim=2)
        refined = self.proj(sampled)
        refined = self.norm(refined)
        refined = self.act(refined)
        return x + self.gamma * refined
