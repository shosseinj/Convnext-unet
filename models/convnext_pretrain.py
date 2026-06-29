import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

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
    def __init__(self, weights_path=None,depth=[3, 3, 9, 3], drop_path_rate=0.1):
        super().__init__()
        
        self.dims = [96, 192, 384, 768]
        self.depths = depth
        
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, self.dims[0], kernel_size=4, stride=4),
            LayerNorm(self.dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm(self.dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample)
        
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        
        self.stages = nn.ModuleList()
        for i in range(4):
            stage_blocks = []
            for j in range(self.depths[i]):
                stage_blocks.append(
                    Block(
                        dim=self.dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=1e-6
                    )
                )
            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.depths[i]
        
        if weights_path:
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
        print(f"✓ Loaded weights from {weights_path}")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
    
    def forward(self, x):
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        return features


# ============================================================================
# BSEI MODULE (Now uses LayerNorm + GELU)
# ============================================================================

class BSEI(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(BSEI, self).__init__()
        # LayerNorm + GELU for consistency
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        self.act2 = nn.GELU()
        
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Edge detection branch
        self.edge_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.act2(self.norm2(self.conv2(x)))
        
        edge = torch.abs(F.conv2d(x, self.edge_conv.weight, padding=1))
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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
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

class ConvNeXtUNet(nn.Module):
    def __init__(self, weights_path=None, num_classes=1, encoder_depth=[3, 3, 9, 3], drop_path_rate=0.1, dropout_rate=0.1):
        super(ConvNeXtUNet, self).__init__()
        
        # Encoder
        self.encoder = ConvNeXtEncoder(
            weights_path=weights_path,
            depth=encoder_depth,
            drop_path_rate=drop_path_rate
        )
        
        dims = [96, 192, 384, 768]
        
        # Bottleneck (LayerNorm + GELU)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(dims[3], dims[3], 3, padding=1),
            LayerNorm(dims[3], eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(dims[3], dims[3], 3, padding=1),
            LayerNorm(dims[3], eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Dropout2d(dropout_rate)
        )
        
        # Decoder (LayerNorm + GELU)
        self.decoder4 = DecoderBlock(dims[3], dims[2], dropout_rate)
        self.bsei4 = BSEI(dims[2] + dims[2], dims[2], dropout_rate)
        
        self.decoder3 = DecoderBlock(dims[2], dims[1], dropout_rate)
        self.bsei3 = BSEI(dims[1] + dims[1], dims[1], dropout_rate)
        
        self.decoder2 = DecoderBlock(dims[1], dims[0], dropout_rate)
        self.bsei2 = BSEI(dims[0] + dims[0], dims[0], dropout_rate)
        
        self.decoder1 = DecoderBlock(dims[0], dims[0], dropout_rate)
        self.bsei1 = BSEI(dims[0], dims[0], dropout_rate)
        
        # Segmentation head (LayerNorm + GELU)
        self.seg_head = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], 3, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Dropout2d(dropout_rate * 0.5),
            nn.Conv2d(dims[0], num_classes, 1)
        )
        
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
        f1, f2, f3, f4 = self.encoder(x)
        
        # Bottleneck
        b = self.bottleneck(f4)
        
        # Decoder with skip connections
        d4 = self.decoder4(b)
        d4 = torch.cat([d4, f3], dim=1)
        d4 = self.bsei4(d4)
        
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, f2], dim=1)
        d3 = self.bsei3(d3)
        
        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, f1], dim=1)
        d2 = self.bsei2(d2)
        
        d1 = self.decoder1(d2)
        d1 = self.bsei1(d1)
        
        # Output
        out = self.seg_head(d1)
        
        if out.shape[-2:] != (352, 352):
            out = F.interpolate(out, size=(352, 352), mode='bilinear', align_corners=True)
        
        return out
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("✓ Encoder frozen")
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("✓ Encoder unfrozen")
    
    def get_encoder_params(self):
        return self.encoder.parameters()
    
    def get_decoder_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'encoder' not in name:
                params.append(param)
        return params

