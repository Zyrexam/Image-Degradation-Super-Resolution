import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):  # Increased default to 64
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class DegradationAwareSR(nn.Module):
    
    def __init__(self, scale=4, d_channels=8, feat_channels=64):  # Changed: 64 default
        super().__init__()
        self.scale = scale
        
        # Lightweight degradation encoder
        self.deg_enc = nn.Sequential(
            nn.Conv2d(3, feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Degradation heads
        self.deg_head1 = nn.Conv2d(feat_channels, d_channels, 3, padding=1)
        self.deg_head2 = nn.Conv2d(feat_channels, d_channels, 3, padding=1)
        
        # Modulation (multiplicative, not additive)
        self.mod = nn.Sequential(
            nn.Conv2d(d_channels, feat_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, 3, 1)
        )
        
        # SR pathway (now at LR resolution)
        self.head = nn.Conv2d(3, feat_channels, 3, padding=1)
        
        # 6 Residual blocks (was 4)
        self.body = nn.Sequential(*[ResidualBlock(feat_channels) for _ in range(6)])
        
        # Upsampling using PixelShuffle (end of network)
        self.upsample = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(feat_channels, 3, 3, padding=1)
        )

    def forward(self, lr1, lr2=None):
        # Shared degradation encoding (at LR resolution)
        f1 = self.deg_enc(lr1)
        d1 = torch.sigmoid(self.deg_head1(f1))
        
        if lr2 is not None:
            f2 = self.deg_enc(lr2)
            d2 = torch.sigmoid(self.deg_head2(f2))
            d_fused = 0.5 * (d1 + d2)
        else:
            d2 = None
            d_fused = d1
        
        # Multiplicative modulation (CRITICAL change)
        mod = torch.tanh(self.mod(d_fused))  # Range [-1, 1]
        
        # Apply modulation: x = lr * (1 + 0.2 * mod)
        # Stronger than additive, still stable
        x = lr1 * (1 + 0.2 * mod)
        x = torch.clamp(x, 0, 1)
        
        # SR pathway (all at LR resolution until PixelShuffle)
        x = self.head(x)
        x = self.body(x)
        x = self.upsample(x)  # PixelShuffle to HR
        
        # Skip connection (bicubic upsampling of LR)
        skip = F.interpolate(lr1, scale_factor=self.scale, mode="bicubic", align_corners=False)
        # sr = x + skip
        sr = x + 0.2 * skip
        sr = torch.clamp(sr, 0, 1)
        
        if lr2 is not None:
            return sr, d1, d2, d_fused
        else:
            return sr, d_fused