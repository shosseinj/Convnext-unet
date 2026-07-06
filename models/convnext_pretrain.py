import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, dim, reduction=8, dropout_rate=0.1):
        super().__init__()
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
                for dilation in (1, 3, 5)
            ]
        )

        self.pool_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, 1, bias=False),
            nn.GELU(),
        )

        self.project = nn.Sequential(
            nn.Conv2d(hidden * 4, dim, 1, bias=False),
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


def resize_like(x, ref):
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(
            x,
            size=ref.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return x
    
class ConvNeXtUNet(nn.Module):
    def __init__(self, weights_path=None, num_classes=1, encoder_depth=[3, 3, 9, 3], drop_path_rate=0.1, dropout_rate=0.1):
        super(ConvNeXtUNet, self).__init__()
        
        # Encoder
        self.encoder = ConvNeXtEncoder(
            weights_path=weights_path,
            depth=encoder_depth,
            drop_path_rate=drop_path_rate, 
            dropout_rate=dropout_rate
        )
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
        
        dims = [96, 192, 384, 768]
        
        self.bottleneck = LiteBottleneck(dims[3], hidden_dim=dims[1], dropout_rate=dropout_rate)
        self.context = MultiScaleContext(dims[3], dropout_rate=dropout_rate)
        
        # Decoder (LayerNorm + GELU)
        self.decoder4 = DecoderBlock(dims[3], dims[2], dropout_rate)
        self.bsei4 = BSEI(dims[2] + dims[2], dims[2], dropout_rate)
        
        self.decoder3 = DecoderBlock(dims[2], dims[1], dropout_rate)
        self.bsei3 = BSEI(dims[1] + dims[1], dims[1], dropout_rate)
        
        self.decoder2 = DecoderBlock(dims[1], dims[0], dropout_rate)
        self.bsei2 = BSEI(dims[0] + dims[0], dims[0], dropout_rate)
        
        self.decoder1 = DecoderBlock(dims[0], dims[0], dropout_rate)
        self.bsei1 = BSEI(dims[0], dims[0], dropout_rate)
        
        self.detail = DetailBranch(out_ch=32, dropout_rate=dropout_rate)
        self.detail_fusion = GatedDetailFusion(
            decoder_ch=dims[0],
            detail_ch=32,
            out_ch=dims[0] + 32,
            dropout_rate=dropout_rate,
        )

        self.final_refine = nn.Sequential(
            SeparableConv2d(96 + 32, 96, 3, padding=1, bias=False),
            LayerNorm(96, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            SeparableConv2d(96, 48, 3, padding=1, bias=False),
            LayerNorm(48, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(48, num_classes, 1)
        )

        self.aux4 = nn.Conv2d(384, num_classes, 1)
        self.aux3 = nn.Conv2d(192, num_classes, 1)
        self.aux2 = nn.Conv2d(96, num_classes, 1)
        
        # Initialize decoder
        self._init_decoder()
    
    def _init_decoder(self):
        for name, module in self.named_modules():
            if 'encoder' in name:
                continue
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Encoder
        encoder_x = (x - self.encoder_mean) / self.encoder_std
        f1, f2, f3, f4 = self.encoder(encoder_x)
        
        # Bottleneck
        b = self.bottleneck(f4)
        b = self.context(b)
        
        # Decoder with skip connections
        d4 = self.decoder4(b)
        d4 = resize_like(d4, f3)
        d4 = torch.cat([d4, f3], dim=1)
        d4 = self.bsei4(d4)
        
        d3 = self.decoder3(d4)
        d3 = resize_like(d3, f2)
        d3 = torch.cat([d3, f2], dim=1)
        d3 = self.bsei3(d3)
        
        d2 = self.decoder2(d3)
        d2 = resize_like(d2, f1)
        d2 = torch.cat([d2, f1], dim=1)
        d2 = self.bsei2(d2)
        detail = self.detail(x)   # H/2 resolution
        d1 = self.decoder1(d2)
        d1 = self.bsei1(d1)
        d1 = self.detail_fusion(d1, detail)
        out_main = self.final_refine(d1)
        out_main = F.interpolate(out_main, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.training:
            aux4 = F.interpolate(self.aux4(d4), size=x.shape[-2:], mode="bilinear", align_corners=False)
            aux3 = F.interpolate(self.aux3(d3), size=x.shape[-2:], mode="bilinear", align_corners=False)
            aux2 = F.interpolate(self.aux2(d2), size=x.shape[-2:], mode="bilinear", align_corners=False)
            return [out_main, aux2, aux3, aux4]

        return out_main
    
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

