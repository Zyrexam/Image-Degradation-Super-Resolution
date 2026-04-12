import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class DegradationAwareSR(nn.Module):
    
    def __init__(self, scale=4, d_channels=8, feat_channels=32):
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
        
        # Simple modulation
        self.mod = nn.Conv2d(d_channels, 3, 1)
        
        # SR pathway (lightweight)
        self.head = nn.Conv2d(3, 32, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(32) for _ in range(4)])  # 4 blocks instead of 8
        self.tail = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, lr1, lr2=None):
        # Shared degradation encoding
        f1 = self.deg_enc(lr1)
        d1 = torch.sigmoid(self.deg_head1(f1))
        
        if lr2 is not None:
            f2 = self.deg_enc(lr2)
            d2 = torch.sigmoid(self.deg_head2(f2))
            d_fused = 0.5 * (d1 + d2)
        else:
            d2 = None
            d_fused = d1
        
        # Simple modulation
        mod = torch.tanh(self.mod(d_fused)) * 0.2
        x = lr1 + mod
        x = torch.clamp(x, 0, 1)
        
        # Upsampling
        x = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        
        # SR network
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        
        # Skip connection
        skip = F.interpolate(lr1, scale_factor=self.scale, mode="bicubic", align_corners=False)
        sr = x + skip
        sr = torch.clamp(sr, 0, 1)
        
        if lr2 is not None:
            return sr, d1, d2, d_fused
        else:
            return sr, d_fused