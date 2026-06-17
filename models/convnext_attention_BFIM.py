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

# ============================================================
# ATTENTION GATE - Updated to use GELU instead of ReLU
# ============================================================
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        
        # CHANGED: ReLU -> GELU
        self.act = nn.GELU() 
        
    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # CHANGED: self.relu -> self.act
        psi = self.act(g1 + x1) 
        psi = self.psi(psi)  
        
        return x * psi



import torch
import torch.nn as nn
import torch.nn.functional as F

class FFD(nn.Module):
    """Frequency Feature Decomposer - Numerically Stable Version"""
    def __init__(self, channels):
        super().__init__()
        self.high_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels), 
            nn.ReLU(inplace=True)
        )
        self.low_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels), 
            nn.ReLU(inplace=True)
        )
        self.channel_mix = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Add input check
        if not torch.isfinite(x).all():
            print("Warning: NaN detected in FFD input")
            # Return identity if input is bad
            return x, x, self.channel_mix(torch.cat([x, x], dim=1))
        
        # FFT with numerical safety
        try:
            freq = torch.fft.rfft2(x, norm='ortho')
        except:
            print("FFT failed, returning identity")
            return x, x, self.channel_mix(torch.cat([x, x], dim=1))
        
        # Safe magnitude computation
        mag = torch.abs(freq).clamp(min=1e-8)  # Prevent division by zero
        
        # Create frequency masks
        mask_h = torch.ones_like(mag)
        cx, cy = H // 2, W // 4
        
        # Ensure indices are within bounds
        cx_start = max(1, cx // 4)
        cx_end = min(H - 1, 3 * cx // 4)
        cy_start = max(1, cy // 4)
        cy_end = min(mag.shape[-1] - 1, 3 * cy // 4)
        
        mask_h[:, :, cx_start:cx_end, cy_start:cy_end] = 0
        mask_l = 1 - mask_h
        
        # Safe inverse FFT
        try:
            f_high = torch.fft.irfft2(freq * mask_h, s=(H, W), norm='ortho')
            f_low = torch.fft.irfft2(freq * mask_l, s=(H, W), norm='ortho')
        except:
            print("Inverse FFT failed, returning identity")
            return x, x, self.channel_mix(torch.cat([x, x], dim=1))
        
        # Clamp to prevent extreme values
        f_high = torch.clamp(f_high, -10, 10)
        f_low = torch.clamp(f_low, -10, 10)
        
        # Replace NaN with 0
        f_high = torch.nan_to_num(f_high, nan=0.0, posinf=10.0, neginf=-10.0)
        f_low = torch.nan_to_num(f_low, nan=0.0, posinf=10.0, neginf=-10.0)
        
        out_h = self.high_conv(f_high)
        out_l = self.low_conv(f_low)
        
        # Final safety check
        out_h = torch.nan_to_num(out_h, nan=0.0)
        out_l = torch.nan_to_num(out_l, nan=0.0)
        
        return out_h, out_l, self.channel_mix(
            torch.cat([out_h, out_l], dim=1))


class BSEI(nn.Module):
    """Boundary-guided Squeeze-Excitation Injection - Numerically Stable"""
    def __init__(self, channels):
        super().__init__()
        kx = torch.tensor(
            [[-1,0,1],[-2,0,2],[-1,0,1]],
            dtype=torch.float32).view(1,1,3,3)
        ky = kx.transpose(-1,-2)
        self.register_buffer('kx', kx.repeat(channels,1,1,1))
        self.register_buffer('ky', ky.repeat(channels,1,1,1))
        self.C = channels
        self.q_proj = nn.Conv2d(channels, channels // 4, 1)
        self.k_proj = nn.Conv2d(channels, channels // 4, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, f_skip, f_dec):
        # Input check
        if not torch.isfinite(f_skip).all() or not torch.isfinite(f_dec).all():
            return f_skip  # Return skip as-is if inputs are bad
        
        # Safe boundary map computation
        gx = F.conv2d(f_skip, self.kx, padding=1, groups=self.C)
        gy = F.conv2d(f_skip, self.ky, padding=1, groups=self.C)
        bmap = torch.sqrt(torch.clamp(gx**2 + gy**2, min=1e-8))  # Prevent sqrt of negative
        
        # Safe normalization
        bmap_max = bmap.amax(dim=(1,2,3), keepdim=True)
        bmap = bmap / (bmap_max + 1e-4)
        bmap = torch.clamp(bmap, 0, 3)
        bmap = torch.nan_to_num(bmap, nan=0.0)
        
        # Weight encoder skip by boundary salience
        f_skip_w = f_skip * (1 + bmap)
        f_skip_w = torch.nan_to_num(f_skip_w, nan=0.0)
        
        # Safe cross-attention
        pool_size = (max(1, f_skip.shape[2] // 4), max(1, f_skip.shape[3] // 4))
        
        Q = self.q_proj(F.adaptive_avg_pool2d(f_dec, pool_size))
        K = self.k_proj(F.adaptive_avg_pool2d(f_skip_w, pool_size))
        V = self.v_proj(F.adaptive_avg_pool2d(f_skip_w, pool_size))
        
        B, Cq, Hp, Wp = Q.shape
        Q = Q.flatten(2).permute(0,2,1)
        K = K.flatten(2).permute(0,2,1)
        V = V.flatten(2).permute(0,2,1)
        
        # Stable attention
        attn = (Q @ K.transpose(-2,-1)) / (Cq**0.5)
        
        # Numerical stability for softmax
        attn = attn - attn.max(dim=-1, keepdim=True)[0]
        attn = torch.nan_to_num(attn, nan=0.0, neginf=-10.0)
        attn = attn.softmax(-1)
        
        ctx = (attn @ V).permute(0,2,1).reshape(B, self.C, Hp, Wp)
        ctx = F.interpolate(self.out_proj(ctx),
                            size=f_skip.shape[2:], mode='bilinear',
                            align_corners=False)
        
        # Safe residual connection
        result = f_skip + self.alpha * ctx
        result = torch.nan_to_num(result, nan=0.0)
        
        return result

class MFAF(nn.Module):
    """Multi-scale Frequency Adaptive Fusion.
    Learns soft per-channel weights between FFD outputs
    and BSEI output, then merges into a single refined feature."""
    def __init__(self, channels):
        super().__init__()
        # gating weights over 3 branches: f_high, f_low, f_bsei
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 3, channels * 3 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 3 // 4, 3),  # 3 branch weights
            nn.Softmax(dim=-1)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
        )

    def forward(self, f_high, f_low, f_bsei):
        combined = torch.cat([f_high, f_low, f_bsei], dim=1)
        w = self.gate(combined)  # (B, 3)
        w0, w1, w2 = w[:,0:1,None,None], w[:,1:2,None,None], w[:,2:3,None,None]
        fused = w0 * f_high + w1 * f_low + w2 * f_bsei
        return self.fuse(fused)


class BFIM(nn.Module):
    """Full Boundary-Frequency Injection Module.
    Drop-in replacement for skip connections in your U-Net.
    Attach one per decoder scale (3 total for a 4-stage encoder)."""
    def __init__(self, channels):
        super().__init__()
        self.ffd  = FFD(channels)
        self.bsei = BSEI(channels)
        self.mfaf = MFAF(channels)

    def forward(self, f_skip, f_dec):
        f_high, f_low, _ = self.ffd(f_skip)
        f_bsei = self.bsei(f_skip, f_dec)
        f_out  = self.mfaf(f_high, f_low, f_bsei)
        return f_out  # replace your current skip in the decoder
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




# ============================================================
# Main Model: ConvNeXt Tiny U-Net with Attention Gates
# ============================================================
class ConvNeXtTinyUNetAttention(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1,
        dims=(96, 192, 384, 768),
        depths=(3, 3, 9, 3),
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
        dec_depths = (2, 2, 2, 2) 
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
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]) if use_batch_norm else nn.Identity(),
            nn.GELU(),
        )

        self.enc1 = ConvNeXtStage(dims[0], depths[0], drop_path_rates=enc_stage_rates[0], dropout=dropout, use_batch_norm=use_batch_norm)
        self.down1 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[1]) if use_batch_norm else nn.Identity(),
        )

        self.enc2 = ConvNeXtStage(dims[1], depths[1], drop_path_rates=enc_stage_rates[1], dropout=dropout, use_batch_norm=use_batch_norm)
        self.down2 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[2]) if use_batch_norm else nn.Identity(),
        )

        self.enc3 = ConvNeXtStage(dims[2], depths[2], drop_path_rates=enc_stage_rates[2], dropout=dropout, use_batch_norm=use_batch_norm)
        self.down3 = nn.Sequential(
            nn.Conv2d(dims[2], dims[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[3]) if use_batch_norm else nn.Identity(),
        )

        self.enc4 = ConvNeXtStage(dims[3], depths[3], drop_path_rates=enc_stage_rates[3], dropout=dropout, use_batch_norm=use_batch_norm)

        # Bottleneck
        self.bottleneck = ConvNeXtStage(dims[3], depths[3], drop_path_rates=enc_stage_rates[3], dropout=dropout, use_batch_norm=use_batch_norm)


        self.bfim3 = BFIM(channels=dims[2])  # skip x3: H/8,  384ch
        self.bfim2 = BFIM(channels=dims[1])  # skip x2: H/4,  192ch
        self.bfim1 = BFIM(channels=dims[0])  # skip x1: H/2,   96ch
        # ====================================================
        # ATTENTION GATES
        # ====================================================
        self.attn3 = AttentionGate(F_g=dims[2], F_l=dims[2], F_int=dims[2]//2)
        self.attn2 = AttentionGate(F_g=dims[1], F_l=dims[1], F_int=dims[1]//2)
        self.attn1 = AttentionGate(F_g=dims[0], F_l=dims[0], F_int=dims[0]//2)

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
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(dims[0]//2, num_classes, kernel_size=1),
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
        # ---------------- Encoder ----------------
        x1 = self.enc1(self.stem(x))      # H/2 -> dims[0]
        x2 = self.enc2(self.down1(x1))    # H/4 -> dims[1]
        x3 = self.enc3(self.down2(x2))    # H/8 -> dims[2]
        x4 = self.enc4(self.down3(x3))    # H/16 -> dims[3]

        # ---------------- Bottleneck ----------------
        b = self.bottleneck(x4)

        # ---------------- Decoder with Attention Gates ----------------
        d3 = self.up3(b)                  # H/8
        x3_bfim = self.bfim3(x3, d3)             # <-- CHANGED (was just x3)
        x3_attn = self.attn3(g=d3, x=x3_bfim)
        d3 = torch.cat([d3, x3_attn], dim=1)  
        d3 = self.reduce3(d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)                 # H/4
        x2_bfim = self.bfim2(x2, d2)             # <-- CHANGED (was just x2)
        x2_attn = self.attn2(g=d2, x=x2_bfim)   
        d2 = torch.cat([d2, x2_attn], dim=1)  
        d2 = self.reduce2(d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)                 # H/2
        x1_bfim = self.bfim1(x1, d1)             # <-- CHANGED (was just x1)
        x1_attn = self.attn1(g=d1, x=x1_bfim)
        d1 = torch.cat([d1, x1_attn], dim=1)  
        d1 = self.reduce1(d1)
        d1 = self.dec1(d1)

        # Recover full resolution (H/2 -> H)
        d1 = self.final_up(d1)            # H

        out = self.seg_head(d1)
        return out