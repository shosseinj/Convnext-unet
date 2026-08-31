import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
from .csaf import CrossScaleAttentionFusion, CrossScaleAttentionFusionV2
from .cross_level_fusion import CrossLevelFusion, CrossLevelFusionV2
from .fafem import FrequencyAwareFeatureEnhancement
from .geometry_conv import GeometryDeformableConv
from pathlib import Path
try:
    from timm.models.layers import trunc_normal_, DropPath
except ModuleNotFoundError:
    trunc_normal_ = nn.init.trunc_normal_

    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor

# ============================================================================
# LAYERNORM (Unified - works for both channels_first and channels_last)
# ============================================================================

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        else:
            raise NotImplementedError


# ============================================================================
# CONVNEXT BLOCK (Encoder - LayerNorm + GELU)
# ============================================================================

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


# ============================================================================
# CONVNEXT ENCODER
# ============================================================================

class MixStyle(nn.Module):
    """Mix instance statistics across batch samples during training only."""

    def __init__(self, probability=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("MixStyle probability must be in [0, 1]")
        if alpha <= 0.0:
            raise ValueError("MixStyle alpha must be positive")
        self.probability = float(probability)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x):
        if not self.training or self.probability == 0.0:
            return x
        if torch.rand((), device=x.device) > self.probability:
            return x
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = (x.var(dim=(2, 3), keepdim=True, unbiased=False) + self.eps).sqrt()
        normalized = (x - mean) / std
        permutation = torch.randperm(x.shape[0], device=x.device)
        mix = torch.distributions.Beta(self.alpha, self.alpha).sample(
            (x.shape[0],)
        ).to(device=x.device, dtype=x.dtype).view(-1, 1, 1, 1)
        mixed_mean = mix * mean + (1.0 - mix) * mean[permutation]
        mixed_std = mix * std + (1.0 - mix) * std[permutation]
        return normalized * mixed_std + mixed_mean


class ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        weights_path=None,
        depth=[3, 3, 9, 3],
        drop_path_rate=0.0,
        dropout_rate=0.25
    ):
        super().__init__()
        
        self.dims = [96, 192, 384, 768]
        self.depths = depth

        # dropout for feature maps
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(3, self.dims[0], kernel_size=4, stride=4),
            LayerNorm(
                self.dims[0],
                eps=1e-6,
                data_format="channels_first"
            )
        )

        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm(
                    self.dims[i],
                    eps=1e-6,
                    data_format="channels_first"
                ),
                nn.Conv2d(
                    self.dims[i],
                    self.dims[i+1],
                    kernel_size=2,
                    stride=2
                ),
            )
            self.downsample_layers.append(downsample)
        
        dp_rates = [
            x.item()
            for x in torch.linspace(
                0,
                drop_path_rate,
                sum(self.depths)
            )
        ]

        cur = 0
        
        self.stages = nn.ModuleList()

        for i in range(4):

            stage_blocks = []

            for j in range(self.depths[i]):

                stage_blocks.append(
                    Block(
                        dim=self.dims[i],
                        drop_path=dp_rates[cur+j],
                        layer_scale_init_value=1e-6
                    )
                )

            self.stages.append(
                nn.Sequential(*stage_blocks)
            )

            cur += self.depths[i]
        

        if weights_path:
            print('loading pretrain weights')
            self._load_weights(weights_path)



    
    def _load_weights(self, weights_path):
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                k = k[9:]
            new_state_dict[k] = v
        
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded weights from {weights_path}")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
    
    def forward(self, x):

        features = []

        for i in range(4):

            x = self.downsample_layers[i](x)

            x = self.stages[i](x)


            # dropout only deeper features
            if i >= 1:
                x = self.dropout(x)


            features.append(x)


        return features



# ============================================================================
# BSEI MODULE (Now uses LayerNorm + GELU)
# ============================================================================

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.0):
        super().__init__()
        self.block = nn.Sequential(
            SeparableConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Dropout2d(dropout_rate),
        )

    def forward(self, x):
        return self.block(x)


class LiteBottleneck(nn.Module):
    def __init__(self, dim, hidden_dim=192, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            LayerNorm(hidden_dim, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            SeparableConv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            LayerNorm(hidden_dim, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(hidden_dim, dim, 1, bias=False),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.gamma * self.net(x)

class BSEI(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(BSEI, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm1 = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        self.act1 = nn.GELU()
        
        self.conv2 = SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        self.act2 = nn.GELU()
        
        self.dropout = nn.Dropout2d(dropout_rate)
        
        self.edge_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False)
        
    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.act2(self.norm2(self.conv2(x)))
        
        edge = torch.abs(self.edge_conv(x))
        edge = torch.sigmoid(edge)
        x = x + edge * x
        return x


class SimpleFusion(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Dropout2d(dropout_rate),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGateSkip(nn.Module):
    """Spatial attention gate followed by matched decoder/skip fusion."""
    def __init__(self, decoder_channels, skip_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        hidden = max(1, out_channels // 2)
        self.decoder_proj = nn.Conv2d(decoder_channels, hidden, 1, bias=False)
        self.skip_proj = nn.Conv2d(skip_channels, hidden, 1, bias=False)
        self.attention = nn.Sequential(nn.GELU(), nn.Conv2d(hidden, 1, 1), nn.Sigmoid())
        self.fuse = SimpleFusion(decoder_channels + skip_channels, out_channels, dropout_rate)

    def forward(self, decoder, skip):
        decoder = resize_like(decoder, skip)
        gate = self.attention(self.decoder_proj(decoder) + self.skip_proj(skip))
        return self.fuse(torch.cat([decoder, skip * gate], dim=1))


class Stage3GatedSkipFusion(nn.Module):
    """Lightweight gate for stage-3 skip before the existing skip fusion."""
    def __init__(self, decoder_channels, skip_channels):
        super().__init__()
        self.decoder_proj = nn.Conv2d(decoder_channels, skip_channels, 1, bias=True)
        self.skip_proj = nn.Conv2d(skip_channels, skip_channels, 1, bias=True)
        nn.init.zeros_(self.decoder_proj.weight)
        nn.init.zeros_(self.decoder_proj.bias)
        nn.init.zeros_(self.skip_proj.weight)
        nn.init.zeros_(self.skip_proj.bias)

    def forward(self, decoder, skip):
        decoder = resize_like(decoder, skip)
        gate = torch.sigmoid(self.decoder_proj(decoder) + self.skip_proj(skip))
        return skip * gate


def channel_shuffle(x, groups):
    """EMCAD channel shuffle for a channels-first tensor."""
    batch_size, channels, height, width = x.shape
    if channels % groups:
        raise ValueError("channels must be divisible by channel-shuffle groups")
    x = x.view(batch_size, groups, channels // groups, height, width)
    x = x.transpose(1, 2).contiguous()
    return x.view(batch_size, channels, height, width)


class MSCBLite(nn.Module):
    """Single residual MSCB using the official EMCAD block topology at C=384."""
    def __init__(self, channels, kernel_sizes=(1, 3, 5), expansion_factor=2):
        super().__init__()
        expanded_channels = channels * expansion_factor
        self.shuffle_groups = math.gcd(expanded_channels, channels)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(channels, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
        )
        self.dwconvs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    expanded_channels,
                    expanded_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    groups=expanded_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU6(inplace=True),
            )
            for kernel_size in kernel_sizes
        )
        self.pconv2 = nn.Sequential(
            nn.Conv2d(expanded_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        expanded = self.pconv1(x)
        multi_scale = sum(branch(expanded) for branch in self.dwconvs)
        refined = self.pconv2(channel_shuffle(multi_scale, self.shuffle_groups))
        return x + refined


class DeformableDepthwiseMSCBBranch(nn.Module):
    """Depthwise deformable MSCB branch with zero-offset ordinary-conv startup."""

    def __init__(self, channels, kernel_size):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size))
        self.offset = nn.Conv2d(channels, 2 * kernel_size * kernel_size, 3, padding=1)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU6(inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, x):
        return self.act(self.norm(deform_conv2d(
            x, self.offset(x), self.weight, bias=None,
            stride=(1, 1), padding=(self.padding, self.padding),
            dilation=(1, 1), mask=None,
        )))


class FrequencyGuidedMSCBLite(MSCBLite):
    """MSCB-lite whose scale branches are guided by the bottleneck FAFEM descriptor."""
    def __init__(self, channels, descriptor_channels, reduction=16):
        super().__init__(channels)
        hidden_channels = max(8, descriptor_channels // (2 * reduction))
        self.guidance_mlp = nn.Sequential(
            nn.Linear(descriptor_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, len(self.dwconvs)),
        )
        self.last_alpha = None

    def reset_guidance(self):
        final_layer = self.guidance_mlp[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def branch_weights(self, descriptor):
        return 3.0 * torch.softmax(self.guidance_mlp(descriptor), dim=1)

    def forward(self, x, frequency_descriptor):
        alpha = self.branch_weights(frequency_descriptor)
        self.last_alpha = alpha.detach()
        expanded = self.pconv1(x)
        branches = [branch(expanded) for branch in self.dwconvs]
        multi_scale = sum(
            alpha[:, index].view(-1, 1, 1, 1) * branch
            for index, branch in enumerate(branches)
        )
        refined = self.pconv2(channel_shuffle(multi_scale, self.shuffle_groups))
        return x + refined


class ResidualFrequencyGuidedMSCBLite(FrequencyGuidedMSCBLite):
    """Bounded residual frequency guidance that starts as ordinary MSCB-lite."""
    def __init__(self, channels, descriptor_channels, reduction=16,
                 guidance_init_std=1e-3, signed_strength=False,
                 initial_guidance_strength=0.0):
        super().__init__(channels, descriptor_channels, reduction)
        if not 0.0 <= initial_guidance_strength <= 0.5:
            raise ValueError("initial_guidance_strength must be in [0, 0.5]")
        self.guidance_strength = nn.Parameter(torch.tensor([initial_guidance_strength]))
        self.guidance_init_std = float(guidance_init_std)
        self.signed_strength = bool(signed_strength)

    def effective_guidance_strength(self):
        if self.signed_strength:
            return 0.5 * torch.tanh(self.guidance_strength)
        return self.guidance_strength.clamp(0.0, 0.5)

    def reset_guidance(self):
        # Strength starts at zero, so the forward path is exactly ordinary MSCB.
        # A positive strength with uniform alpha keeps the initial output identical
        # to ordinary MSCB while letting the MLP receive gradients immediately.
        final_layer = self.guidance_mlp[-1]
        if self.guidance_init_std:
            nn.init.normal_(final_layer.weight, mean=0.0, std=self.guidance_init_std)
        else:
            nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, x, frequency_descriptor):
        alpha = self.branch_weights(frequency_descriptor)
        self.last_alpha = alpha.detach()
        expanded = self.pconv1(x)
        branches = [branch(expanded) for branch in self.dwconvs]
        ordinary_sum = sum(branches)
        guided_sum = sum(
            alpha[:, index].view(-1, 1, 1, 1) * branch
            for index, branch in enumerate(branches)
        )
        strength = self.effective_guidance_strength()
        multi_scale = ordinary_sum + strength * (guided_sum - ordinary_sum)
        refined = self.pconv2(channel_shuffle(multi_scale, self.shuffle_groups))
        return x + refined


class F4F3ContextResidualFrequencyGuidedMSCBLite(ResidualFrequencyGuidedMSCBLite):
    """Residual FG-MSCB guided jointly by FAFEM f4 and raw Stage-3 f3 context."""

    def __init__(self, channels, frequency_descriptor_channels, reduction=16,
                 guidance_init_std=0.0, signed_strength=False,
                 initial_guidance_strength=0.0):
        combined_channels = frequency_descriptor_channels + channels
        super().__init__(channels, combined_channels, reduction,
                         guidance_init_std=guidance_init_std,
                         signed_strength=signed_strength,
                         initial_guidance_strength=initial_guidance_strength)
        self.frequency_descriptor_channels = frequency_descriptor_channels
        self.local_descriptor_channels = channels
        self.guidance_mlp = nn.Sequential(
            nn.Linear(combined_channels, 48), nn.GELU(), nn.Linear(48, 3)
        )
        self.last_descriptor_shapes = None
        self.reset_guidance()

    def combined_descriptor(self, frequency_descriptor, f3):
        local = torch.flatten(F.adaptive_avg_pool2d(f3, 1), start_dim=1)
        combined = torch.cat((frequency_descriptor, local), dim=1)
        self.last_descriptor_shapes = {
            "f4_frequency_descriptor": tuple(frequency_descriptor.shape),
            "f3_local_descriptor": tuple(local.shape),
            "combined_descriptor": tuple(combined.shape),
        }
        return combined

    def forward(self, x, frequency_descriptor, f3):
        return super().forward(x, self.combined_descriptor(frequency_descriptor, f3))


class DeformableResidualFrequencyGuidedMSCBLite(ResidualFrequencyGuidedMSCBLite):
    """Exp.45 residual FG-MSCB with deformable sampling in each spatial branch."""

    def __init__(self, channels, descriptor_channels, reduction=16,
                 guidance_init_std=1e-3, signed_strength=False,
                 initial_guidance_strength=0.0):
        super().__init__(
            channels, descriptor_channels, reduction,
            guidance_init_std=guidance_init_std,
            signed_strength=signed_strength,
            initial_guidance_strength=initial_guidance_strength,
        )
        self.dwconvs = nn.ModuleList(
            DeformableDepthwiseMSCBBranch(self.pconv1[0].out_channels, kernel_size)
            for kernel_size in (1, 3, 5)
        )

    def reset_deformable_branches(self):
        for branch in self.dwconvs:
            branch.reset_parameters()


class PartialDeformableResidualFrequencyGuidedMSCBLite(ResidualFrequencyGuidedMSCBLite):
    """Exp.45 FG-MSCB with deformable sampling only in the 3x3 and 5x5 branches."""

    def __init__(self, channels, descriptor_channels, reduction=16,
                 guidance_init_std=1e-3, signed_strength=False,
                 initial_guidance_strength=0.0):
        super().__init__(
            channels, descriptor_channels, reduction,
            guidance_init_std=guidance_init_std,
            signed_strength=signed_strength,
            initial_guidance_strength=initial_guidance_strength,
        )
        expanded_channels = self.pconv1[0].out_channels
        self.dwconvs = nn.ModuleList((
            self.dwconvs[0],
            DeformableDepthwiseMSCBBranch(expanded_channels, 3),
            DeformableDepthwiseMSCBBranch(expanded_channels, 5),
        ))

    def reset_deformable_branches(self):
        for branch in self.dwconvs[1:]:
            branch.reset_parameters()


class LKALiteStage3(nn.Module):
    """Identity-safe large-kernel attention for the 384-channel Stage-3 skip."""
    def __init__(self, channels):
        super().__init__()
        self.dwconv5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.dwconv7_dilated = nn.Conv2d(
            channels, channels, 7, padding=9, dilation=3, groups=channels
        )
        self.pwconv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        attention = self.pwconv(self.dwconv7_dilated(self.dwconv5(x)))
        return x + self.gamma * (x * attention)


# ============================================================================
# DECODER BLOCK (LayerNorm + GELU)
# ============================================================================

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout_rate)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


# ============================================================================
# CONVNEXT U-NET (Unified: LayerNorm + GELU everywhere)
# ============================================================================
class DetailBranch(nn.Module):
    def __init__(self, out_ch=32, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Dropout2d(dropout_rate * 0.5),
            SeparableConv2d(32, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout_rate * 0.5),
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleContext(nn.Module):
    def __init__(self, dim, reduction=8, dropout_rate=0.1, dilations=(1, 3, 5)):
        super().__init__()
        if not dilations or len(dilations) > 5 or any(value <= 0 for value in dilations):
            raise ValueError("MSC dilations must contain 1..5 positive integers")
        self.dilations = tuple(dilations)
        hidden = dim // reduction

        self.reduce = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            LayerNorm(hidden, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
        )

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        hidden,
                        hidden,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        groups=hidden,
                        bias=False,
                    ),
                    LayerNorm(hidden, eps=1e-6, data_format="channels_first"),
                    nn.GELU(),
                )
                for dilation in self.dilations
            ]
        )

        self.pool_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, 1, bias=False),
            nn.GELU(),
        )

        self.project = nn.Sequential(
            nn.Conv2d(hidden * (len(self.dilations) + 1), dim, 1, bias=False),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Dropout2d(dropout_rate),
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        reduced = self.reduce(x)
        pooled = self.pool_proj(reduced)
        pooled = F.interpolate(
            pooled,
            size=reduced.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        context = torch.cat([branch(reduced) for branch in self.branches] + [pooled], dim=1)
        return x + self.gamma * self.project(context)


class GatedDetailFusion(nn.Module):
    def __init__(self, decoder_ch=96, detail_ch=32, out_ch=128, dropout_rate=0.1):
        super().__init__()
        self.detail_proj = nn.Sequential(
            SeparableConv2d(detail_ch, detail_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(detail_ch),
            nn.GELU(),
            nn.Dropout2d(dropout_rate * 0.5),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(decoder_ch + detail_ch, detail_ch, 1),
            nn.GELU(),
            nn.Conv2d(detail_ch, detail_ch, 1),
            nn.Sigmoid(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_ch + detail_ch, out_ch, 1, bias=False),
            LayerNorm(out_ch, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Dropout2d(dropout_rate * 0.5),
        )

    def forward(self, decoder_feat, detail_feat):
        if detail_feat.shape[-2:] != decoder_feat.shape[-2:]:
            detail_feat = F.interpolate(
                detail_feat,
                size=decoder_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        detail_feat = self.detail_proj(detail_feat)
        gate = self.gate(torch.cat([decoder_feat, detail_feat], dim=1))
        detail_feat = detail_feat * gate
        return self.fuse(torch.cat([decoder_feat, detail_feat], dim=1))


class AdditionDetailFusion(nn.Module):
    def __init__(self, decoder_ch, detail_ch, out_ch, dropout_rate=0.1):
        super().__init__()
        self.detail_proj = nn.Conv2d(detail_ch, decoder_ch, 1, bias=False)
        self.refine = SimpleFusion(decoder_ch, out_ch, dropout_rate)

    def forward(self, decoder_feat, detail_feat):
        detail_feat = resize_like(detail_feat, decoder_feat)
        return self.refine(decoder_feat + self.detail_proj(detail_feat))


class AttentionDetailFusion(nn.Module):
    def __init__(self, decoder_ch, detail_ch, out_ch, dropout_rate=0.1):
        super().__init__()
        self.detail_proj = nn.Conv2d(detail_ch, decoder_ch, 1, bias=False)
        self.gate = nn.Sequential(nn.Conv2d(decoder_ch * 2, decoder_ch, 1), nn.Sigmoid())
        self.refine = SimpleFusion(decoder_ch, out_ch, dropout_rate)

    def forward(self, decoder_feat, detail_feat):
        detail_feat = resize_like(self.detail_proj(detail_feat), decoder_feat)
        gate = self.gate(torch.cat([decoder_feat, detail_feat], dim=1))
        return self.refine(decoder_feat + gate * detail_feat)


def resize_like(x, ref):
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(
            x,
            size=ref.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return x


class TorchvisionFeatureEncoder(nn.Module):
    """Four-scale ImageNet encoder backed by an explicit local weight file."""
    def __init__(self, backbone, weights_path):
        super().__init__()
        from torchvision.models import efficientnet_b0, resnet34

        if weights_path is None or not Path(weights_path).is_file():
            raise FileNotFoundError(f"Required local pretrained weights missing for {backbone}: {weights_path}")
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if backbone == "resnet34":
            model = resnet34(weights=None)
            model.load_state_dict(state, strict=True)
            self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
            self.stages = nn.ModuleList([model.layer1, model.layer2, model.layer3, model.layer4])
            self.channels = (64, 128, 256, 512)
            self.kind = backbone
        elif backbone == "efficientnet_b0":
            model = efficientnet_b0(weights=None)
            model.load_state_dict(state, strict=True)
            self.features = model.features
            self.channels = (24, 40, 112, 320)
            self.kind = backbone
        else:
            raise ValueError(f"Unsupported torchvision backbone: {backbone}")

    def forward(self, x):
        if self.kind == "resnet34":
            x = self.stem(x)
            outputs = []
            for stage in self.stages:
                x = stage(x); outputs.append(x)
            return tuple(outputs)
        outputs = []
        for index, block in enumerate(self.features):
            x = block(x)
            if index in {2, 3, 5, 7}:
                outputs.append(x)
        return tuple(outputs)
    
class ConvNeXtUNet(nn.Module):
    def __init__(self, weights_path=None, num_classes=1, encoder_depth=[3, 3, 9, 3],
                 drop_path_rate=0.1, dropout_rate=0.1, enable_msc=True,
                 skip_mode="normal", detail_channels=0, enable_gdf=False,
                 deep_supervision_heads=0, msc_dilations=(1, 3, 5),
                 detail_fusion_mode=None, backbone="convnext_tiny",
                 enable_csaf=False, enable_fafem=False, csaf_version="v1",
                 fafem_stage1=False, fafem_stage2=False, fafem_stage3=False,
                 enable_gated_skip_stage3=False,
                 enable_cross_level_fusion=False,
                 cross_level_fusion_version="v1",
                 enable_geometry_conv_stage3=False,
                 decoder_highres_width=96,
                 enable_mscb_lite_stage3=False,
                 enable_fg_mscb_lite_stage3=False,
                 enable_residual_fg_mscb_lite_stage3=False,
                 residual_fg_mscb_guidance_init_std=1e-3,
                 residual_fg_mscb_signed_strength=False,
                 residual_fg_mscb_initial_strength=0.0,
                 enable_deformable_residual_fg_mscb_lite_stage3=False,
                 enable_residual_fg_mscb_all_skips=False,
                 enable_partial_deformable_residual_fg_mscb_lite_stage3=False,
                 enable_f4_f3_context_guided_mscb_lite_stage3=False,
                 enable_mixstyle_stage1_stage2=False,
                 enable_lka_lite_stage3=False,
                 enable_mscb_lite_stage2=False,
                 enable_mscb_lite_stage1=False):
        super(ConvNeXtUNet, self).__init__()
        if skip_mode not in {"normal", "attention_gate", "bsei"}:
            raise ValueError(f"Unsupported skip_mode: {skip_mode}")
        if detail_channels < 0:
            raise ValueError("detail_channels must be non-negative")
        if enable_gdf and detail_channels == 0:
            raise ValueError("GDF requires detail_channels > 0")
        if detail_fusion_mode is None:
            detail_fusion_mode = "gdf" if enable_gdf else ("concatenation" if detail_channels else "none")
        if detail_fusion_mode not in {"none", "addition", "concatenation", "attention_fusion", "gdf"}:
            raise ValueError(f"Unsupported detail_fusion_mode: {detail_fusion_mode}")
        if detail_fusion_mode != "none" and detail_channels == 0:
            raise ValueError("Detail fusion requires detail_channels > 0")
        if deep_supervision_heads not in {0, 1, 2, 3}:
            raise ValueError("deep_supervision_heads must be between 0 and 3")
        if backbone not in {"resnet34", "efficientnet_b0", "convnext_tiny"}:
            raise ValueError(f"Unsupported backbone: {backbone}")
        if csaf_version not in {"v1", "v2"}:
            raise ValueError(f"Unsupported CSAF version: {csaf_version}")
        if cross_level_fusion_version not in {"v1", "v2"}:
            raise ValueError(
                f"Unsupported Cross-Level Fusion version: {cross_level_fusion_version}"
            )
        if decoder_highres_width < 96:
            raise ValueError("decoder_highres_width must be at least 96")
        if (enable_fg_mscb_lite_stage3 or enable_residual_fg_mscb_lite_stage3 or
                enable_deformable_residual_fg_mscb_lite_stage3 or
                enable_residual_fg_mscb_all_skips or
                enable_partial_deformable_residual_fg_mscb_lite_stage3 or
                enable_f4_f3_context_guided_mscb_lite_stage3):
            if not enable_fafem:
                raise ValueError("FG-MSCB requires bottleneck FAFEM")
            if (enable_mscb_lite_stage3 or enable_mscb_lite_stage2 or
                    enable_mscb_lite_stage1):
                raise ValueError("FG-MSCB cannot coexist with ordinary MSCB-lite blocks")
        if sum((enable_fg_mscb_lite_stage3, enable_residual_fg_mscb_lite_stage3,
                enable_deformable_residual_fg_mscb_lite_stage3,
                enable_residual_fg_mscb_all_skips,
                enable_partial_deformable_residual_fg_mscb_lite_stage3,
                enable_f4_f3_context_guided_mscb_lite_stage3)) > 1:
            raise ValueError("Only one frequency-guided MSCB-lite variant may be enabled")
        self.variant_config = {
            "enable_msc": bool(enable_msc),
            "skip_mode": skip_mode,
            "detail_channels": int(detail_channels),
            "enable_gdf": bool(enable_gdf),
            "deep_supervision_heads": int(deep_supervision_heads),
            "msc_dilations": tuple(msc_dilations),
            "detail_fusion_mode": detail_fusion_mode,
            "backbone": backbone,
            "enable_csaf": bool(enable_csaf),
            "enable_fafem": bool(enable_fafem),
            "csaf_version": csaf_version,
            "fafem_stage1": bool(fafem_stage1),
            "fafem_stage2": bool(fafem_stage2),
            "fafem_stage3": bool(fafem_stage3),
            "enable_gated_skip_stage3": bool(enable_gated_skip_stage3),
            "enable_cross_level_fusion": bool(enable_cross_level_fusion),
            "cross_level_fusion_version": cross_level_fusion_version,
            "enable_geometry_conv_stage3": bool(enable_geometry_conv_stage3),
            "decoder_highres_width": int(decoder_highres_width),
            "enable_mscb_lite_stage3": bool(enable_mscb_lite_stage3),
            "enable_fg_mscb_lite_stage3": bool(enable_fg_mscb_lite_stage3),
            "enable_residual_fg_mscb_lite_stage3": bool(enable_residual_fg_mscb_lite_stage3),
            "residual_fg_mscb_guidance_init_std": float(residual_fg_mscb_guidance_init_std),
            "residual_fg_mscb_signed_strength": bool(residual_fg_mscb_signed_strength),
            "residual_fg_mscb_initial_strength": float(residual_fg_mscb_initial_strength),
            "enable_deformable_residual_fg_mscb_lite_stage3": bool(enable_deformable_residual_fg_mscb_lite_stage3),
            "enable_residual_fg_mscb_all_skips": bool(enable_residual_fg_mscb_all_skips),
            "enable_partial_deformable_residual_fg_mscb_lite_stage3": bool(enable_partial_deformable_residual_fg_mscb_lite_stage3),
            "enable_f4_f3_context_guided_mscb_lite_stage3": bool(enable_f4_f3_context_guided_mscb_lite_stage3),
            "enable_mixstyle_stage1_stage2": bool(enable_mixstyle_stage1_stage2),
            "enable_lka_lite_stage3": bool(enable_lka_lite_stage3),
            "enable_mscb_lite_stage2": bool(enable_mscb_lite_stage2),
            "enable_mscb_lite_stage1": bool(enable_mscb_lite_stage1),
        }
        
        # Encoder
        if backbone == "convnext_tiny":
            self.encoder = ConvNeXtEncoder(
                weights_path=weights_path,
                depth=encoder_depth,
                drop_path_rate=drop_path_rate,
                dropout_rate=dropout_rate
            )
            encoder_channels = (96, 192, 384, 768)
        else:
            self.encoder = TorchvisionFeatureEncoder(backbone, weights_path)
            encoder_channels = self.encoder.channels
        self.register_buffer(
            "encoder_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "encoder_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        
        dims = [96, 192, 384, encoder_channels[3]]

        if enable_csaf and csaf_version == "v2":
            rng_state = torch.get_rng_state()
            self.csaf = nn.ModuleList(
                (
                    CrossScaleAttentionFusionV2(
                        encoder_channels, (0, 1), 0, encoder_channels[0]
                    ),
                    CrossScaleAttentionFusionV2(
                        encoder_channels, (0, 1, 2), 1, encoder_channels[1]
                    ),
                    CrossScaleAttentionFusionV2(
                        encoder_channels, (1, 2, 3), 2, encoder_channels[2]
                    ),
                )
            )
            torch.set_rng_state(rng_state)
        elif enable_csaf:
            self.csaf = nn.ModuleList(
                CrossScaleAttentionFusion(
                    encoder_channels, index, encoder_channels[index]
                )
                for index in range(3)
            )
        else:
            self.csaf = None
        
        self.bottleneck = LiteBottleneck(dims[3], hidden_dim=dims[1], dropout_rate=dropout_rate)
        self.context = (MultiScaleContext(dims[3], dropout_rate=dropout_rate,
                                          dilations=msc_dilations)
                        if enable_msc else nn.Identity())
        
        # Decoder (LayerNorm + GELU)
        fusion = BSEI if skip_mode == "bsei" else SimpleFusion
        self.decoder4 = DecoderBlock(dims[3], dims[2], dropout_rate)
        self.bsei4 = (AttentionGateSkip(dims[2], encoder_channels[2], dims[2], dropout_rate)
                      if skip_mode == "attention_gate" else
                      fusion(dims[2] + encoder_channels[2], dims[2], dropout_rate))
        
        self.decoder3 = DecoderBlock(dims[2], dims[1], dropout_rate)
        self.bsei3 = (AttentionGateSkip(dims[1], encoder_channels[1], dims[1], dropout_rate)
                      if skip_mode == "attention_gate" else
                      fusion(dims[1] + encoder_channels[1], dims[1], dropout_rate))
        
        self.decoder2 = DecoderBlock(dims[1], dims[0], dropout_rate)
        self.bsei2 = (AttentionGateSkip(dims[0], encoder_channels[0], dims[0], dropout_rate)
                      if skip_mode == "attention_gate" else
                      fusion(dims[0] + encoder_channels[0], dims[0], dropout_rate))
        
        self.decoder1 = DecoderBlock(dims[0], decoder_highres_width, dropout_rate)
        self.bsei1 = (SimpleFusion(decoder_highres_width, decoder_highres_width, dropout_rate)
                      if skip_mode == "attention_gate" else fusion(decoder_highres_width, decoder_highres_width, dropout_rate))

        self.detail_branch = DetailBranch(detail_channels, dropout_rate) if detail_channels else None
        if detail_fusion_mode == "gdf":
            self.detail_fusion = GatedDetailFusion(dims[0], detail_channels, dims[0], dropout_rate)
        elif detail_fusion_mode == "addition":
            self.detail_fusion = AdditionDetailFusion(dims[0], detail_channels, dims[0], dropout_rate)
        elif detail_fusion_mode == "attention_fusion":
            self.detail_fusion = AttentionDetailFusion(dims[0], detail_channels, dims[0], dropout_rate)
        elif detail_fusion_mode == "concatenation":
            self.detail_fusion = SimpleFusion(dims[0] + detail_channels, dims[0], dropout_rate)
        else:
            self.detail_fusion = None

        auxiliary_channels = (dims[0], dims[1], dims[2])
        self.auxiliary_heads = nn.ModuleList(
            [nn.Conv2d(auxiliary_channels[index], num_classes, 1)
             for index in range(deep_supervision_heads)]
        )
        
        final_refine_hidden = max(48, decoder_highres_width // 2)
        self.final_refine = nn.Sequential(
            SeparableConv2d(decoder_highres_width, decoder_highres_width, 3, padding=1, bias=False),
            LayerNorm(decoder_highres_width, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            SeparableConv2d(decoder_highres_width, final_refine_hidden, 3, padding=1, bias=False),
            LayerNorm(final_refine_hidden, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(final_refine_hidden, num_classes, 1)
        )

        # Keep common baseline initialization reproducible: constructing this
        # optional module must not advance the RNG used by existing modules.
        rng_state = torch.get_rng_state()
        self.fafem = (
            FrequencyAwareFeatureEnhancement(encoder_channels[3])
            if enable_fafem else None
        )
        self.fafem_stage1 = (
            FrequencyAwareFeatureEnhancement(encoder_channels[0])
            if fafem_stage1 else None
        )
        self.fafem_stage2 = (
            FrequencyAwareFeatureEnhancement(encoder_channels[1])
            if fafem_stage2 else None
        )
        self.fafem_stage3 = (
            FrequencyAwareFeatureEnhancement(encoder_channels[2])
            if fafem_stage3 else None
        )
        self.mixstyle_stage1 = MixStyle(probability=0.5, alpha=0.1) if enable_mixstyle_stage1_stage2 else None
        self.mixstyle_stage2 = MixStyle(probability=0.5, alpha=0.1) if enable_mixstyle_stage1_stage2 else None
        self.gated_skip_stage3 = (
            Stage3GatedSkipFusion(dims[2], encoder_channels[2])
            if enable_gated_skip_stage3 else None
        )
        self.cross_level_fusion = (
            (CrossLevelFusionV2(encoder_channels[:3], fusion_channels=64)
             if cross_level_fusion_version == "v2" else
             CrossLevelFusion(encoder_channels[:3], fusion_channels=encoder_channels[0]))
            if enable_cross_level_fusion else None
        )
        self.geometry_conv_stage3 = (
            GeometryDeformableConv(encoder_channels[2])
            if enable_geometry_conv_stage3 else None
        )
        self.mscb_lite_stage3 = (
            MSCBLite(dims[2]) if enable_mscb_lite_stage3 else None
        )
        self.fg_mscb_lite_stage3 = (
            FrequencyGuidedMSCBLite(dims[2], encoder_channels[3] * 2)
            if enable_fg_mscb_lite_stage3 else None
        )
        self.residual_fg_mscb_lite_stage3 = (
            ResidualFrequencyGuidedMSCBLite(
                dims[2], encoder_channels[3] * 2,
                guidance_init_std=residual_fg_mscb_guidance_init_std,
                signed_strength=residual_fg_mscb_signed_strength,
                initial_guidance_strength=residual_fg_mscb_initial_strength,
            )
            if enable_residual_fg_mscb_lite_stage3 else None
        )
        self.deformable_residual_fg_mscb_lite_stage3 = (
            DeformableResidualFrequencyGuidedMSCBLite(
                dims[2], encoder_channels[3] * 2,
                guidance_init_std=residual_fg_mscb_guidance_init_std,
                signed_strength=residual_fg_mscb_signed_strength,
                initial_guidance_strength=residual_fg_mscb_initial_strength,
            ) if enable_deformable_residual_fg_mscb_lite_stage3 else None
        )
        self.partial_deformable_residual_fg_mscb_lite_stage3 = (
            PartialDeformableResidualFrequencyGuidedMSCBLite(
                dims[2], encoder_channels[3] * 2,
                guidance_init_std=residual_fg_mscb_guidance_init_std,
                signed_strength=residual_fg_mscb_signed_strength,
                initial_guidance_strength=residual_fg_mscb_initial_strength,
            ) if enable_partial_deformable_residual_fg_mscb_lite_stage3 else None
        )
        self.f4_f3_context_guided_mscb_lite_stage3 = (
            F4F3ContextResidualFrequencyGuidedMSCBLite(
                dims[2], encoder_channels[3] * 2,
                guidance_init_std=residual_fg_mscb_guidance_init_std,
                signed_strength=residual_fg_mscb_signed_strength,
                initial_guidance_strength=residual_fg_mscb_initial_strength,
            ) if enable_f4_f3_context_guided_mscb_lite_stage3 else None
        )
        self.residual_fg_mscb_lite_stage1 = (
            ResidualFrequencyGuidedMSCBLite(
                encoder_channels[0], encoder_channels[3] * 2,
                guidance_init_std=residual_fg_mscb_guidance_init_std,
                signed_strength=residual_fg_mscb_signed_strength,
                initial_guidance_strength=residual_fg_mscb_initial_strength,
            ) if enable_residual_fg_mscb_all_skips else None
        )
        self.residual_fg_mscb_lite_stage2 = (
            ResidualFrequencyGuidedMSCBLite(
                encoder_channels[1], encoder_channels[3] * 2,
                guidance_init_std=residual_fg_mscb_guidance_init_std,
                signed_strength=residual_fg_mscb_signed_strength,
                initial_guidance_strength=residual_fg_mscb_initial_strength,
            ) if enable_residual_fg_mscb_all_skips else None
        )
        self.residual_fg_mscb_lite_stage3_skip = (
            ResidualFrequencyGuidedMSCBLite(
                encoder_channels[2], encoder_channels[3] * 2,
                guidance_init_std=residual_fg_mscb_guidance_init_std,
                signed_strength=residual_fg_mscb_signed_strength,
                initial_guidance_strength=residual_fg_mscb_initial_strength,
            ) if enable_residual_fg_mscb_all_skips else None
        )
        self.mscb_lite_stage2 = (
            MSCBLite(dims[1]) if enable_mscb_lite_stage2 else None
        )
        self.mscb_lite_stage1 = (
            MSCBLite(dims[0]) if enable_mscb_lite_stage1 else None
        )
        self.lka_lite_stage3 = (
            LKALiteStage3(encoder_channels[2]) if enable_lka_lite_stage3 else None
        )
        torch.set_rng_state(rng_state)
        
        # Initialize decoder
        self._init_decoder()
        if self.fg_mscb_lite_stage3 is not None:
            self.fg_mscb_lite_stage3.reset_guidance()
        if self.residual_fg_mscb_lite_stage3 is not None:
            self.residual_fg_mscb_lite_stage3.reset_guidance()
        if self.deformable_residual_fg_mscb_lite_stage3 is not None:
            self.deformable_residual_fg_mscb_lite_stage3.reset_deformable_branches()
            self.deformable_residual_fg_mscb_lite_stage3.reset_guidance()
        if self.partial_deformable_residual_fg_mscb_lite_stage3 is not None:
            self.partial_deformable_residual_fg_mscb_lite_stage3.reset_deformable_branches()
            self.partial_deformable_residual_fg_mscb_lite_stage3.reset_guidance()
        if self.f4_f3_context_guided_mscb_lite_stage3 is not None:
            self.f4_f3_context_guided_mscb_lite_stage3.reset_guidance()
        for module in (
                self.residual_fg_mscb_lite_stage1,
                self.residual_fg_mscb_lite_stage2,
                self.residual_fg_mscb_lite_stage3_skip):
            if module is not None:
                module.reset_guidance()
    
    def _init_decoder(self):
        for name, module in self.named_modules():
            if 'encoder' in name:
                continue
            if self.variant_config["csaf_version"] == "v2" and name.startswith("csaf."):
                continue
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        original_input = x
        # Encoder
        encoder_x = (x - self.encoder_mean) / self.encoder_std
        f1, f2, f3, f4 = self.encoder(encoder_x)
        if self.mixstyle_stage1 is not None:
            f1 = self.mixstyle_stage1(f1)
        if self.mixstyle_stage2 is not None:
            f2 = self.mixstyle_stage2(f2)
        raw_f3_for_context = f3
        if self.fafem_stage1 is not None:
            f1 = self.fafem_stage1(f1)
        if self.fafem_stage2 is not None:
            f2 = self.fafem_stage2(f2)
        if self.fafem_stage3 is not None:
            f3 = self.fafem_stage3(f3)
        fafem_applied = False
        frequency_descriptor = None
        frequency_guided_mscb = (
            self.fg_mscb_lite_stage3 or self.residual_fg_mscb_lite_stage3 or
            self.deformable_residual_fg_mscb_lite_stage3 or
            self.partial_deformable_residual_fg_mscb_lite_stage3 or
            self.f4_f3_context_guided_mscb_lite_stage3
        )
        frequency_guided_skip_modules = (
            self.residual_fg_mscb_lite_stage1,
            self.residual_fg_mscb_lite_stage2,
            self.residual_fg_mscb_lite_stage3_skip,
        )
        requires_frequency_descriptor = (
            frequency_guided_mscb is not None or
            any(module is not None for module in frequency_guided_skip_modules)
        )
        if self.fafem is not None and self.variant_config["csaf_version"] == "v2":
            if requires_frequency_descriptor:
                f4, frequency_descriptor = self.fafem.forward_with_frequency_descriptor(f4)
            else:
                f4 = self.fafem(f4)
            fafem_applied = True
        if self.csaf is not None:
            encoder_features = (f1, f2, f3, f4)
            f1, f2, f3 = (
                fusion(encoder_features) for fusion in self.csaf
            )
        if self.cross_level_fusion is not None:
            f1, f2, f3 = self.cross_level_fusion(f1, f2, f3)
        if self.geometry_conv_stage3 is not None:
            f3 = self.geometry_conv_stage3(f3)
        if self.lka_lite_stage3 is not None:
            f3 = self.lka_lite_stage3(f3)
        if self.fafem is not None and not fafem_applied:
            if requires_frequency_descriptor:
                f4, frequency_descriptor = self.fafem.forward_with_frequency_descriptor(f4)
            else:
                f4 = self.fafem(f4)

        if any(module is not None for module in frequency_guided_skip_modules):
            if frequency_descriptor is None:
                raise RuntimeError("Skip FG-MSCB requires the bottleneck FAFEM descriptor")
            if self.residual_fg_mscb_lite_stage1 is not None:
                f1 = self.residual_fg_mscb_lite_stage1(f1, frequency_descriptor)
            if self.residual_fg_mscb_lite_stage2 is not None:
                f2 = self.residual_fg_mscb_lite_stage2(f2, frequency_descriptor)
            if self.residual_fg_mscb_lite_stage3_skip is not None:
                f3 = self.residual_fg_mscb_lite_stage3_skip(f3, frequency_descriptor)
        
        # Bottleneck
        b = self.bottleneck(f4)
        b = self.context(b)
        
        # Decoder with skip connections
        d4 = self.decoder4(b)
        d4 = resize_like(d4, f3)
        if self.gated_skip_stage3 is not None:
            f3 = self.gated_skip_stage3(d4, f3)
        d4 = (self.bsei4(d4, f3) if self.variant_config["skip_mode"] == "attention_gate"
              else self.bsei4(torch.cat([d4, f3], dim=1)))
        if self.mscb_lite_stage3 is not None:
            d4 = self.mscb_lite_stage3(d4)
        if frequency_guided_mscb is not None:
            if frequency_descriptor is None:
                raise RuntimeError("FG-MSCB requires the bottleneck FAFEM descriptor")
            if self.f4_f3_context_guided_mscb_lite_stage3 is not None:
                d4 = frequency_guided_mscb(d4, frequency_descriptor, raw_f3_for_context)
            else:
                d4 = frequency_guided_mscb(d4, frequency_descriptor)
        
        d3 = self.decoder3(d4)
        d3 = resize_like(d3, f2)
        d3 = (self.bsei3(d3, f2) if self.variant_config["skip_mode"] == "attention_gate"
              else self.bsei3(torch.cat([d3, f2], dim=1)))
        if self.mscb_lite_stage2 is not None:
            d3 = self.mscb_lite_stage2(d3)
        
        d2 = self.decoder2(d3)
        d2 = resize_like(d2, f1)
        d2 = (self.bsei2(d2, f1) if self.variant_config["skip_mode"] == "attention_gate"
              else self.bsei2(torch.cat([d2, f1], dim=1)))
        if self.mscb_lite_stage1 is not None:
            d2 = self.mscb_lite_stage1(d2)

        d1 = self.decoder1(d2)
        d1 = self.bsei1(d1)
        if self.detail_branch is not None:
            detail = self.detail_branch(original_input)
            if self.variant_config["detail_fusion_mode"] != "concatenation":
                d1 = self.detail_fusion(d1, detail)
            else:
                detail = resize_like(detail, d1)
                d1 = self.detail_fusion(torch.cat([d1, detail], dim=1))
        out_main = self.final_refine(d1)
        out_main = F.interpolate(out_main, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if not self.auxiliary_heads:
            return out_main
        auxiliary_features = (d2, d3, d4)
        auxiliary_outputs = [
            F.interpolate(head(feature), size=x.shape[-2:], mode="bilinear", align_corners=False)
            for head, feature in zip(self.auxiliary_heads, auxiliary_features)
        ]
        return (out_main, *auxiliary_outputs)
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen")
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen")
    
    def get_encoder_params(self):
        return self.encoder.parameters()
    
    def get_decoder_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'encoder' not in name:
                params.append(param)
        return params

