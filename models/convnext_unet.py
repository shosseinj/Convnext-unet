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
# Enhanced ConvNeXt Block with Dropout and Regularization
# ============================================================
class BlockReLU(nn.Module):
    def __init__(
        self, 
        dim, 
        drop_path=0.1,
        dropout=0.1,
        layer_scale_init_value=1e-6,
        use_batch_norm=True
    ):
        super().__init__()

        # Depthwise convolution with smaller kernel
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=5,  # Reduced from 7 to 5 for medical images
            padding=2,
            groups=dim,
            bias=False
        )
        
        # Optional BatchNorm after depthwise conv
        self.use_bn = use_batch_norm
        if use_batch_norm:
            self.norm = nn.BatchNorm2d(dim)
        
        # Pointwise convolutions
        self.pwconv1 = nn.Linear(dim, 4 * dim, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)  # Dropout after activation
        
        self.pwconv2 = nn.Linear(4 * dim, dim, bias=False)
        self.dropout2 = nn.Dropout(dropout)  # Dropout after second conv
        
        # Layer scale (learnable parameter)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        
        # Depthwise convolution
        x = self.dwconv(x)
        
        if self.use_bn:
            x = self.norm(x)
        
        # Reshape for linear layers
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        
        # Pointwise convolution 1 + activation + dropout
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.dropout1(x)
        
        # Pointwise convolution 2 + dropout
        x = self.pwconv2(x)
        x = self.dropout2(x)
        
        # Layer scale
        x = self.gamma * x
        
        # Reshape back
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        # Stochastic depth + residual connection
        x = identity + self.drop_path(x)
        
        return x


# ============================================================
# ConvNeXt Stage with Linear Drop Path Rate
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
            # Linearly increase drop path rate from 0 to max_drop_path
            drop_path_rates = [x.item() for x in torch.linspace(0, 0.2, depth)]
        
        # Make sure we have enough drop path rates
        if len(drop_path_rates) < depth:
            # Extend with the last value if not enough
            drop_path_rates = drop_path_rates + [drop_path_rates[-1]] * (depth - len(drop_path_rates))
        
        self.blocks = nn.Sequential(
            *[BlockReLU(
                dim=dim,
                drop_path=drop_path_rates[i],
                dropout=dropout,
                use_batch_norm=use_batch_norm
            ) for i in range(depth)]
        )

    def forward(self, x):
        return self.blocks(x)


# ============================================================
# Enhanced ConvNeXt Tiny U-Net
# ============================================================
class ConvNeXtTinyUNet(nn.Module):

    def __init__(
        self,
        in_chans=3,
        num_classes=1,
        dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2),
        dropout=0.1,
        drop_path_rate=0.2,
        use_batch_norm=True
    ):
        super().__init__()
        
        self.dims = dims
        self.depths = depths
        
        # FIXED: Calculate drop path rates properly
        # Total blocks in encoder (excluding bottleneck)
        enc_blocks = sum(depths)  # 2 + 2 + 4 + 2 = 10
        # Total blocks in decoder (excluding bottleneck)
        dec_blocks = sum(depths)  # 2 + 4 + 2 + 2 = 10 (reversed order)
        # Bottleneck blocks
        bn_blocks = depths[-1]  # 2
        
        # Total blocks that need drop path rates
        total_blocks = enc_blocks + dec_blocks + bn_blocks
        
        # Generate all drop path rates
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        # Split rates: encoder | bottleneck | decoder
        enc_rates = dp_rates[:enc_blocks]  # First 10
        bn_rates = dp_rates[enc_blocks:enc_blocks + bn_blocks]  # Next 2
        dec_rates = dp_rates[enc_blocks + bn_blocks:]  # Last 10
        
        # Split encoder rates per stage (in original order)
        idx = 0
        enc_stage_rates = []
        for d in depths:
            enc_stage_rates.append(enc_rates[idx:idx+d])
            idx += d
        
        # Split decoder rates per stage (in REVERSED order for decoder)
        # Decoder stages: dec3, dec2, dec1 (which use depths[2], depths[1], depths[0])
        idx = 0
        dec_stage_rates = []
        decoder_depths = list(reversed(depths))  # [2, 4, 2, 2]
        for d in decoder_depths:
            end_idx = idx + d
            dec_stage_rates.append(dec_rates[idx:end_idx])
            idx = end_idx
        
        # Debug: Print the split
        print(f"Encoder stage blocks: {depths}")
        print(f"Decoder stage blocks: {decoder_depths}")
        print(f"Encoder stage rates lengths: {[len(r) for r in enc_stage_rates]}")
        print(f"Decoder stage rates lengths: {[len(r) for r in dec_stage_rates]}")
        print(f"Bottleneck rates length: {len(bn_rates)}")

        # ====================================================
        # Encoder
        # ====================================================
        
        # Stem with smaller kernels and batch norm
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0]//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]//2) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.5),  # Lighter dropout in stem
            nn.Conv2d(dims[0]//2, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        self.enc1 = ConvNeXtStage(
            dims[0], depths[0], 
            drop_path_rates=enc_stage_rates[0],
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[1]) if use_batch_norm else nn.Identity(),
        )

        self.enc2 = ConvNeXtStage(
            dims[1], depths[1],
            drop_path_rates=enc_stage_rates[1],
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[2]) if use_batch_norm else nn.Identity(),
        )

        self.enc3 = ConvNeXtStage(
            dims[2], depths[2],
            drop_path_rates=enc_stage_rates[2],
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(dims[2], dims[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[3]) if use_batch_norm else nn.Identity(),
        )

        self.enc4 = ConvNeXtStage(
            dims[3], depths[3],
            drop_path_rates=enc_stage_rates[3],
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )

        # ====================================================
        # Bottleneck with higher dropout
        # ====================================================
        bottleneck_dropout = dropout * 1.5  # Higher dropout in bottleneck
        
        self.bottleneck = ConvNeXtStage(
            dims[3], depths[3],
            drop_path_rates=bn_rates,
            dropout=bottleneck_dropout,
            use_batch_norm=use_batch_norm
        )

        # ====================================================
        # Decoder
        # ====================================================

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(dims[3], dims[2], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dims[2]) if use_batch_norm else nn.Identity(),
        )

        self.reduce3 = nn.Sequential(
            nn.Conv2d(dims[2] * 2, dims[2], kernel_size=1, bias=False),
            nn.BatchNorm2d(dims[2]) if use_batch_norm else nn.Identity(),
        )

        # dec3 uses depths[2] = 4, which is decoder_depths[1]
        self.dec3 = ConvNeXtStage(
            dims[2], depths[2],
            drop_path_rates=dec_stage_rates[1],  # Index 1 for depths[2]
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(dims[2], dims[1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dims[1]) if use_batch_norm else nn.Identity(),
        )

        self.reduce2 = nn.Sequential(
            nn.Conv2d(dims[1] * 2, dims[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(dims[1]) if use_batch_norm else nn.Identity(),
        )

        # dec2 uses depths[1] = 2, which is decoder_depths[2]
        self.dec2 = ConvNeXtStage(
            dims[1], depths[1],
            drop_path_rates=dec_stage_rates[2],  # Index 2 for depths[1]
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(dims[1], dims[0], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dims[0]) if use_batch_norm else nn.Identity(),
        )

        self.reduce1 = nn.Sequential(
            nn.Conv2d(dims[0] * 2, dims[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(dims[0]) if use_batch_norm else nn.Identity(),
        )

        # dec1 uses depths[0] = 2, which is decoder_depths[3]
        self.dec1 = ConvNeXtStage(
            dims[0], depths[0],
            drop_path_rates=dec_stage_rates[3],  # Index 3 for depths[0]
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )

        # ====================================================
        # Final upsampling (2 stages to recover from stem's 4x downsampling)
        # ====================================================
        
        self.final_up1 = nn.Sequential(
            nn.ConvTranspose2d(dims[0], dims[0]//2, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dims[0]//2) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        
        self.final_up2 = nn.Sequential(
            nn.ConvTranspose2d(dims[0]//2, dims[0]//4, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dims[0]//4) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # Segmentation head with dropout
        self.seg_head = nn.Sequential(
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(dims[0]//4, num_classes, kernel_size=1),
        )

        self._init_weights()

    # ========================================================
    # Weight Initialization
    # ========================================================
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ========================================================
    # Forward Pass
    # ========================================================
    def forward(self, x):
        # ---------------- Encoder ----------------
        x1 = self.enc1(self.stem(x))      # H/4 -> dims[0]
        x2 = self.enc2(self.down1(x1))    # H/8 -> dims[1]
        x3 = self.enc3(self.down2(x2))    # H/16 -> dims[2]
        x4 = self.enc4(self.down3(x3))    # H/32 -> dims[3]

        # ---------------- Bottleneck ----------------
        b = self.bottleneck(x4)

        # ---------------- Decoder ----------------
        d3 = self.up3(b)                  # H/16
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.reduce3(d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)                 # H/8
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.reduce2(d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)                 # H/4
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.reduce1(d1)
        d1 = self.dec1(d1)

        # Recover full resolution (H/4 -> H/2 -> H)
        d1 = self.final_up1(d1)           # H/2
        d1 = self.final_up2(d1)           # H

        out = self.seg_head(d1)
        return out


# # ============================================================
# # Usage and Testing
# # ============================================================
# if __name__ == "__main__":
#     # Create model with regularization
#     model = ConvNeXtTinyUNet(
#         in_chans=3,
#         num_classes=1,
#         dims=(64, 128, 256, 512),
#         depths=(2, 2, 4, 2),
#         dropout=0.1,
#         drop_path_rate=0.2,
#         use_batch_norm=True
#     )

#     # Test with random input
#     x = torch.randn(2, 3, 256, 256)
#     y = model(x)

#     print("\nInput shape:", x.shape)
#     print("Output shape:", y.shape)
    
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     print(f"Total Parameters: {total_params:,}")
#     print(f"Trainable Parameters: {trainable_params:,}")
    
#     # Test with training mode (dropout active)
#     model.train()
#     y_train = model(x)
    
#     # Test with eval mode (dropout inactive)
#     model.eval()
#     y_eval = model(x)
    
#     print(f"\nOutput stats (train mode): min={y_train.min():.3f}, max={y_train.max():.3f}, mean={y_train.mean():.3f}")
#     print(f"Output stats (eval mode): min={y_eval.min():.3f}, max={y_eval.max():.3f}, mean={y_eval.mean():.3f}")