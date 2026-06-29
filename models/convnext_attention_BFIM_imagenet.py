import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# DropPath (Stochastic Depth) for regularization
# ============================================================
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output





import torch
import torch.nn as nn
import torch.nn.functional as F





class BSEI(nn.Module):
    """
    Boundary-Semantic Enhancement with Interaction (BSEI).
    FFTEnhance removed — did not improve performance.
    ASG lives outside this class in the decoder (parallel, trainable).
    """
    def __init__(self, channels, pool_size=4):
        super().__init__()

        self.channels  = channels
        self.pool_size = pool_size

        # 1. Edge enhancement
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # 2. Pooled cross-attention (decoder=q, skip=k/v)
        attn_dim = max(channels // 8, 16)

        self.q    = nn.Conv2d(channels, attn_dim, 1, bias=False)
        self.k    = nn.Conv2d(channels, attn_dim, 1, bias=False)
        self.v    = nn.Conv2d(channels, channels,  1, bias=False)

        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # Zero-init → identity at start
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, skip, dec):
        # ---- 1. Edge enhancement ----------------------------------------
        edge      = self.edge_conv(skip)
        skip_edge = skip * (1.0 + edge)

        # ---- 2. Pooled cross-attention -----------------------------------
        q = F.adaptive_avg_pool2d(self.q(dec),       (self.pool_size, self.pool_size))
        k = F.adaptive_avg_pool2d(self.k(skip_edge), (self.pool_size, self.pool_size))
        v = F.adaptive_avg_pool2d(self.v(skip_edge), (self.pool_size, self.pool_size))

        B, Cq, Hp, Wp = q.shape

        q = q.flatten(2).transpose(1, 2)   # (B, S, attn_dim)
        k = k.flatten(2).transpose(1, 2)   # (B, S, attn_dim)
        v = v.flatten(2).transpose(1, 2)   # (B, S, C)

        attn = torch.matmul(q, k.transpose(-1, -2)) * (Cq ** -0.5)
        attn = F.softmax(attn, dim=-1)

        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).reshape(B, self.channels, Hp, Wp)
        ctx = self.proj(ctx)

        ctx = F.interpolate(ctx, size=skip.shape[2:],
                            mode='bilinear', align_corners=False)

        # ---- 3. Output --------------------------------------------------
        return skip_edge + self.alpha * ctx



# ============================================================
# ConvNeXt Block (Renamed from BlockReLU, uses GELU)
# ============================================================
class ConvNeXtBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        drop_path=0.1,
        dropout=0.1,
        layer_scale_init_value=1e-6,
        use_batch_norm=True
    ):
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim, bias=False
        )
        
        self.use_bn = use_batch_norm
        if use_batch_norm:
            self.norm = nn.BatchNorm2d(dim)
        
        self.pwconv1 = nn.Linear(dim, 4 * dim, bias=False)
        self.act = nn.GELU()  
        self.dropout1 = nn.Dropout(dropout)  
        
        self.pwconv2 = nn.Linear(4 * dim, dim, bias=False)
        self.dropout2 = nn.Dropout(dropout)  
        
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        
        if self.use_bn:
            x = self.norm(x)
        
        x = x.permute(0, 2, 3, 1) 
        
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.dropout1(x)
        
        x = self.pwconv2(x)
        x = self.dropout2(x)
        
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2) 
        
        x = identity + self.drop_path(x)
        return x

# ============================================================
# ConvNeXt Stage
# ============================================================
class ConvNeXtStage(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        drop_path_rates=None,
        dropout=0.1,
        use_batch_norm=True
    ):
        super().__init__()
        
        if drop_path_rates is None:
            drop_path_rates = [x.item() for x in torch.linspace(0, 0.2, depth)]
        
        if len(drop_path_rates) < depth:
            drop_path_rates = drop_path_rates + [drop_path_rates[-1]] * (depth - len(drop_path_rates))
        
        # Updated to use ConvNeXtBlock
        self.blocks = nn.Sequential(
            *[ConvNeXtBlock(
                dim=dim,
                drop_path=drop_path_rates[i],
                dropout=dropout,
                use_batch_norm=use_batch_norm
            ) for i in range(depth)]
        )

    def forward(self, x):
        return self.blocks(x)


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


class ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        weights_path=None,
        depth=[3, 3, 9, 3],
        drop_path_rate=0.1,
        dropout=0.1
    ):
        super().__init__()
        
        self.dims = [96, 192, 384, 768]
        self.depths = depth

        # dropout for feature maps
        self.dropout = nn.Dropout2d(p=dropout)
        
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


            # dropout only deeper features
            if i >= 2:
                x = self.dropout(x)


            features.append(x)


        return features




class ConvNeXtTinyUNetAttention(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1,
        dims=(96, 192, 384, 768),
        depths=(2,2,4,2),
        dropout=0.1,
        drop_path_rate=0.2,
        use_batch_norm=True
    ):
        super().__init__()
        
        self.dims = dims
        self.depths = depths
        
        # 1. Calculate drop path rates for Encoder
        enc_blocks = sum(depths)
        enc_rates = [x.item() for x in torch.linspace(0, drop_path_rate, enc_blocks)]
        idx = 0
        enc_stage_rates = []
        for d in depths:
            enc_stage_rates.append(enc_rates[idx:idx+d])
            idx += d
            
        # 2. Calculate drop path rates for Decoder
        dec_depths = (1,1,1,1) 
        dec_blocks = sum(dec_depths)
        dec_rates = [x.item() for x in torch.linspace(0, drop_path_rate, dec_blocks)]
        idx = 0
        dec_stage_rates = []
        for d in dec_depths:
            dec_stage_rates.append(dec_rates[idx:idx+d])
            idx += d

        # ====================================================
        # Encoder (Stem + 4 stages)
        # ====================================================



        self.bsei3 = BSEI(dims[2])
        self.bsei2 = BSEI(dims[1])
        self.bsei1 = BSEI(dims[0])  # H/2 is large, keep pools small

        # self.hca1 = HCABlock(dims[0])
        # self.hca2 = HCABlock(dims[1])
        # self.hca3 = HCABlock(dims[2])

        # ====================================================
        # Decoder (CHANGED: Upsample + Conv instead of ConvTranspose2d)
        # ====================================================

        # Decoder Stage 3
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(dims[3], dims[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dims[2]) if use_batch_norm else nn.Identity(),
            nn.GELU(),
        )
        self.reduce3 = nn.Sequential(
            nn.Conv2d(dims[2] * 2, dims[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dims[2]) if use_batch_norm else nn.Identity(),
            nn.GELU(),
        )
        self.dec3 = ConvNeXtStage(dims[2], dec_depths[2], drop_path_rates=dec_stage_rates[2], dropout=dropout, use_batch_norm=use_batch_norm)

        # Decoder Stage 2
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(dims[2], dims[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dims[1]) if use_batch_norm else nn.Identity(),
            nn.GELU(),
        )
        self.reduce2 = nn.Sequential(
            nn.Conv2d(dims[1] * 2, dims[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dims[1]) if use_batch_norm else nn.Identity(),
            nn.GELU(),
        )
        self.dec2 = ConvNeXtStage(dims[1], dec_depths[1], drop_path_rates=dec_stage_rates[1], dropout=dropout, use_batch_norm=use_batch_norm)

        # Decoder Stage 1
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(dims[1], dims[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]) if use_batch_norm else nn.Identity(),
            nn.GELU(),
        )
        self.reduce1 = nn.Sequential(
            nn.Conv2d(dims[0] * 2, dims[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]) if use_batch_norm else nn.Identity(),
            nn.GELU(),
        )
        self.dec1 = ConvNeXtStage(dims[0], dec_depths[0], drop_path_rates=dec_stage_rates[0], dropout=dropout, use_batch_norm=use_batch_norm)

        # ====================================================
        # Final upsampling 
        # ====================================================
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(dims[0], dims[0]//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]//2) if use_batch_norm else nn.Identity(),
            nn.GELU(),
        )

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Dropout2d( 0.05),
            nn.Conv2d(dims[0]//2, num_classes, kernel_size=1),
        )
        # self.att3 = AttentionGate(dims[2], dims[2], dims[2]//2)
        # self.att2 = AttentionGate(dims[1], dims[1], dims[1]//2)
        # self.att1 = AttentionGate(dims[0], dims[0], dims[0]//2)
        self.encoder = ConvNeXtEncoder(
            weights_path=None,
            depth=[3,3,9,3],
            drop_path_rate=drop_path_rate
        )
        

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):


        x1, x2, x3, x4 = self.encoder(x)
        # ---------------- Bottleneck ----------------
        # b = self.bottleneck(x4)

        # ---------------- Decoder with Attention Gates ----------------
        d3 = self.up3(x4)                  # H/8

        x3_bsei = self.bsei3(
            x3,
            d3
        )

        d3 = torch.cat([d3, x3_bsei], dim=1)

        d3 = self.reduce3(d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)                 # H/4

        x2_bsei = self.bsei2(
            x2,
            d2
        )

        d2 = torch.cat([d2, x2_bsei], dim=1)


        d2 = self.reduce2(d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)                 # H/2
        
        x1_bsei = self.bsei1(
            x1,
            d1
        )

        d1 = torch.cat([d1, x1_bsei], dim=1)

        d1 = self.reduce1(d1)
        d1 = self.dec1(d1)

        d1 = self.final_up(d1)            # H

        out = self.seg_head(d1)
        return out