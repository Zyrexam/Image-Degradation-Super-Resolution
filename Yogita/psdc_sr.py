"""
PSDC-SR: Probabilistic Spatial Degradation Consistency for Super-Resolution
============================================================================
Implementation of the full pipeline including:
  - Shared CNN encoder
  - Probabilistic spatial degradation head (mu, logvar)
  - FiLM-conditioned SR backbone (generator G)
  - Cross-conditioning: SR1 = G(LR1, z2), SR2 = G(LR2, z1)
  - Degradation network D (cycle consistency)
  - Full loss suite: L_rec, L_KL, L_cycle, L_smooth

Usage:
    model = PSDCSRModel(scale=4)
    out   = model(lr1, lr2, hr)   # returns dict of losses + SR outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1.  Utility blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Standard residual block used across encoder and decoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class UpBlock(nn.Module):
    """2× upsample via pixel shuffle."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ---------------------------------------------------------------------------
# 2.  Shared Encoder  E
# ---------------------------------------------------------------------------

class SharedEncoder(nn.Module):
    """
    Encodes LR image → spatial feature map F  (B, C, H/4, W/4).

    Architecture: 3 strided conv blocks (each halves spatial resolution).
    Can be swapped for a UNet encoder or ESRGAN RRDBNet trunk if desired.
    """

    def __init__(self, in_channels: int = 3, base_ch: int = 64, num_res: int = 4):
        super().__init__()
        self.head = nn.Conv2d(in_channels, base_ch, 3, 1, 1)

        # Three stride-2 downsampling stages → H/8 total; we stop at H/4
        self.down1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, 2, 1),    # H/2
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, 2, 1),  # H/4
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.res_blocks = nn.Sequential(*[ResBlock(base_ch * 2) for _ in range(num_res)])
        self.out_channels = base_ch * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        return x   # (B, C, H/4, W/4)


# ---------------------------------------------------------------------------
# 3.  Probabilistic Degradation Head  Hd
# ---------------------------------------------------------------------------

class DegradationHead(nn.Module):
    """
    Maps feature map F → (mu, logvar) both in (B, Cd, H/4, W/4).

    The spatial resolution means each pixel position gets its own
    degradation distribution — this is what makes the model "spatial".
    """

    def __init__(self, in_channels: int, latent_channels: int = 32):
        super().__init__()
        self.mu_head     = nn.Conv2d(in_channels, latent_channels, 3, 1, 1)
        self.logvar_head = nn.Conv2d(in_channels, latent_channels, 3, 1, 1)

    def forward(self, f: torch.Tensor):
        mu     = self.mu_head(f)
        logvar = self.logvar_head(f)
        return mu, logvar


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Standard reparameterization trick.  z = mu + sigma * eps."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# ---------------------------------------------------------------------------
# 4.  FiLM Conditioning Layer
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    Modulates feature map F with degradation code z:

        F' = gamma(z) * F + beta(z)

    gamma, beta are predicted by 1×1 convolutions applied to z,
    then broadcast over H×W.
    """

    def __init__(self, feature_channels: int, latent_channels: int):
        super().__init__()
        self.gamma_conv = nn.Conv2d(latent_channels, feature_channels, 1)
        self.beta_conv  = nn.Conv2d(latent_channels, feature_channels, 1)

    def forward(self, f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # z may be smaller spatially than f — upsample if needed
        if z.shape[-2:] != f.shape[-2:]:
            z = F.interpolate(z, size=f.shape[-2:], mode='bilinear', align_corners=False)
        gamma = self.gamma_conv(z)
        beta  = self.beta_conv(z)
        return gamma * f + beta


# ---------------------------------------------------------------------------
# 5.  SR Generator / Backbone  G
# ---------------------------------------------------------------------------

class SRGenerator(nn.Module):
    """
    Conditioned super-resolution backbone.

    Takes LR features (from encoder) and a degradation code z,
    applies FiLM conditioning after each residual block,
    then upsamples to HR resolution.

    scale=4 → two 2× UpBlocks
    scale=2 → one 2× UpBlock
    """

    def __init__(
        self,
        feature_channels: int,
        latent_channels: int  = 32,
        num_res: int          = 8,
        scale: int            = 4,
        out_channels: int     = 3,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList([ResBlock(feature_channels) for _ in range(num_res)])
        self.film_layers = nn.ModuleList(
            [FiLMLayer(feature_channels, latent_channels) for _ in range(num_res)]
        )

        # Build upsampling path
        up_blocks = []
        ch = feature_channels
        for _ in range(scale // 2):     # 4× = two 2× steps
            up_blocks.append(UpBlock(ch))
        self.up_blocks = nn.Sequential(*up_blocks)

        self.tail = nn.Conv2d(feature_channels, out_channels, 3, 1, 1)

    def forward(self, f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        f : (B, C, H/4, W/4)   — feature from encoder
        z : (B, Cd, H/4, W/4)  — sampled degradation code
        returns SR : (B, 3, H*scale, W*scale)
        """
        x = f
        for res, film in zip(self.res_blocks, self.film_layers):
            x = res(x)
            x = film(x, z)
        x = self.up_blocks(x)
        return self.tail(x)


# ---------------------------------------------------------------------------
# 6.  Degradation Network  D  (cycle branch)
# ---------------------------------------------------------------------------

class DegradationNetwork(nn.Module):
    """
    Learns to re-degrade an SR output back to LR space.
    Used for cycle-consistency:  LR_recon = D(SR_main).

    Simple stride-based downsampler so the gradient flows cleanly.
    """

    def __init__(self, in_channels: int = 3, scale: int = 4):
        super().__init__()
        layers = []
        ch = 32
        layers += [nn.Conv2d(in_channels, ch, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
        steps = scale // 2
        for _ in range(steps):
            layers += [
                nn.Conv2d(ch, ch * 2, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch *= 2
        layers += [nn.Conv2d(ch, in_channels, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, sr: torch.Tensor) -> torch.Tensor:
        return self.net(sr)


# ---------------------------------------------------------------------------
# 7.  Loss Functions
# ---------------------------------------------------------------------------

def kl_consistency(mu1, logvar1, mu2, logvar2) -> torch.Tensor:
    """
    Symmetric KL between two Gaussian distributions.
    KL(N1||N2) + KL(N2||N1) — encourages consistent but not identical codes.

    KL(N(mu1, s1) || N(mu2, s2)) =
        log(s2/s1) + (s1² + (mu1-mu2)²) / (2*s2²) - 0.5
    """
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    kl_1_2 = (logvar2 - logvar1 + (var1 + (mu1 - mu2).pow(2)) / (var2 + 1e-8) - 1) * 0.5
    kl_2_1 = (logvar1 - logvar2 + (var2 + (mu2 - mu1).pow(2)) / (var1 + 1e-8) - 1) * 0.5
    return (kl_1_2 + kl_2_1).mean()


def spatial_smoothness(z: torch.Tensor) -> torch.Tensor:
    """Total-variation loss on the spatial degradation map."""
    dx = z[:, :, :, 1:] - z[:, :, :, :-1]
    dy = z[:, :, 1:, :] - z[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


# ---------------------------------------------------------------------------
# 8.  Full PSDC-SR Model
# ---------------------------------------------------------------------------

class PSDCSRModel(nn.Module):
    """
    Wraps all components into one nn.Module.

    Forward pass returns a dict with:
        'sr1'      : super-resolved output conditioned on z2 (cross)
        'sr2'      : super-resolved output conditioned on z1 (cross)
        'sr_main'  : super-resolved output conditioned on z1 (standard, for cycle)
        'lr_recon' : re-degraded LR from sr_main  (cycle branch)
        'mu1', 'logvar1', 'mu2', 'logvar2' : distributions
        'z1', 'z2' : sampled codes
        'losses'   : dict of individual losses + total
    """

    def __init__(
        self,
        scale: int           = 4,
        base_ch: int         = 64,
        latent_ch: int       = 32,
        num_enc_res: int     = 4,
        num_gen_res: int     = 8,
        w_kl: float          = 0.3,
        w_cycle: float       = 0.2,
        w_smooth: float      = 0.05,
    ):
        super().__init__()
        self.scale    = scale
        self.w_kl     = w_kl
        self.w_cycle  = w_cycle
        self.w_smooth = w_smooth

        # Shared encoder
        self.encoder = SharedEncoder(base_ch=base_ch, num_res=num_enc_res)
        feat_ch = self.encoder.out_channels

        # Degradation head
        self.deg_head = DegradationHead(feat_ch, latent_ch)

        # SR generator
        self.generator = SRGenerator(feat_ch, latent_ch, num_gen_res, scale)

        # Cycle degradation network
        self.deg_net = DegradationNetwork(scale=scale)

    def forward(self, lr1: torch.Tensor, lr2: torch.Tensor, hr: torch.Tensor):
        # ---- Encode ----
        f1 = self.encoder(lr1)
        f2 = self.encoder(lr2)

        # ---- Probabilistic degradation heads ----
        mu1, logvar1 = self.deg_head(f1)
        mu2, logvar2 = self.deg_head(f2)

        z1 = reparameterize(mu1, logvar1)
        z2 = reparameterize(mu2, logvar2)

        # ---- Cross-conditioned SR (KEY NOVELTY) ----
        sr1 = self.generator(f1, z2)   # LR1 content + LR2 degradation code
        sr2 = self.generator(f2, z1)   # LR2 content + LR1 degradation code

        # ---- Standard (non-cross) for cycle ----
        sr_main = self.generator(f1, z1)

        # ---- Cycle: re-degrade SR back to LR space ----
        lr_recon = self.deg_net(sr_main)

        # ---- Losses ----
        # Resize HR to match SR if needed (safety)
        hr_h, hr_w = sr1.shape[-2], sr1.shape[-1]
        if hr.shape[-2:] != (hr_h, hr_w):
            hr = F.interpolate(hr, size=(hr_h, hr_w), mode='bilinear', align_corners=False)

        # Resize LR1 to match lr_recon if needed
        lr1_h, lr1_w = lr_recon.shape[-2], lr_recon.shape[-1]
        if lr1.shape[-2:] != (lr1_h, lr1_w):
            lr1_matched = F.interpolate(lr1, size=(lr1_h, lr1_w), mode='bilinear', align_corners=False)
        else:
            lr1_matched = lr1

        l_rec    = F.l1_loss(sr1, hr) + F.l1_loss(sr2, hr)
        l_kl     = kl_consistency(mu1, logvar1, mu2, logvar2)
        l_cycle  = F.l1_loss(lr_recon, lr1_matched)
        l_smooth = spatial_smoothness(z1) + spatial_smoothness(z2)

        l_total  = l_rec + self.w_kl * l_kl + self.w_cycle * l_cycle + self.w_smooth * l_smooth

        return {
            'sr1'      : sr1,
            'sr2'      : sr2,
            'sr_main'  : sr_main,
            'lr_recon' : lr_recon,
            'mu1': mu1, 'logvar1': logvar1,
            'mu2': mu2, 'logvar2': logvar2,
            'z1' : z1,  'z2' : z2,
            'losses': {
                'total'  : l_total,
                'rec'    : l_rec,
                'kl'     : l_kl,
                'cycle'  : l_cycle,
                'smooth' : l_smooth,
            },
        }


# ---------------------------------------------------------------------------
# 9.  Minimal Training Loop
# ---------------------------------------------------------------------------

class Trainer:
    """
    Minimal trainer.  Plug in your DataLoader and call .fit().

    DataLoader should yield dicts with keys 'hr' (B,3,H,W).
    LR1 and LR2 are generated on-the-fly from HR inside the loop.
    """

    def __init__(
        self,
        model: PSDCSRModel,
        lr: float           = 2e-4,
        device: str         = 'cuda',
        log_every: int      = 100,
    ):
        self.model     = model.to(device)
        self.device    = device
        self.log_every = log_every
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

        # Simple cosine LR scheduler — adjust T_max to your epoch count
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

    @staticmethod
    def degrade(hr: torch.Tensor, scale: int = 4) -> torch.Tensor:
        """
        Minimal stochastic degradation for training.
        In practice replace with Real-ESRGAN's degradation pipeline.

        Pipeline:  Gaussian blur  →  downsample  →  Gaussian noise  →  JPEG
        """
        b, c, h, w = hr.shape
        lh, lw = h // scale, w // scale

        # Random blur kernel (sigma 0.2 – 3.0)
        sigma   = 0.2 + torch.rand(1).item() * 2.8
        ks      = 2 * int(3 * sigma) + 1
        padding = ks // 2
        weight  = Trainer._gaussian_kernel(ks, sigma).to(hr.device)
        weight  = weight.expand(c, 1, ks, ks)
        blurred = F.conv2d(hr, weight, padding=padding, groups=c)

        # Bicubic downsample
        lr = F.interpolate(blurred, size=(lh, lw), mode='bicubic', align_corners=False)

        # Additive noise (std 0 – 25/255)
        noise_std = torch.rand(1).item() * (25 / 255)
        lr = lr + torch.randn_like(lr) * noise_std

        return lr.clamp(0, 1)

    @staticmethod
    def _gaussian_kernel(ks: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(ks, dtype=torch.float32) - ks // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return torch.outer(g, g).unsqueeze(0).unsqueeze(0)

    def step(self, hr: torch.Tensor):
        hr   = hr.to(self.device)
        lr1  = self.degrade(hr, self.model.scale)
        lr2  = self.degrade(hr, self.model.scale)   # independent degradation

        self.optimizer.zero_grad()
        out  = self.model(lr1, lr2, hr)
        loss = out['losses']['total']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {k: v.item() for k, v in out['losses'].items()}

    def fit(self, loader, num_epochs: int = 100):
        step = 0
        for epoch in range(num_epochs):
            for batch in loader:
                hr     = batch['hr'] if isinstance(batch, dict) else batch[0]
                losses = self.step(hr)
                step  += 1
                if step % self.log_every == 0:
                    msg = ' | '.join(f'{k}: {v:.4f}' for k, v in losses.items())
                    print(f'[Epoch {epoch+1:03d} | Step {step:05d}]  {msg}')
            self.scheduler.step()


# ---------------------------------------------------------------------------
# 10. Quick sanity check  (run: python psdc_sr.py)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    model = PSDCSRModel(scale=4, base_ch=64, latent_ch=32, num_enc_res=4, num_gen_res=8)
    model = model.to(device)

    # Fake batch: HR 128×128 → LR should be 32×32
    hr  = torch.rand(2, 3, 128, 128).to(device)
    lr1 = torch.rand(2, 3,  32,  32).to(device)
    lr2 = torch.rand(2, 3,  32,  32).to(device)

    out = model(lr1, lr2, hr)

    print('SR1 shape :', out['sr1'].shape)         # expect (2, 3, 128, 128)
    print('SR2 shape :', out['sr2'].shape)
    print('LR_recon  :', out['lr_recon'].shape)    # expect (2, 3, 32, 32)
    print('z1 shape  :', out['z1'].shape)          # expect (2, 32, 8, 8)
    print()
    for k, v in out['losses'].items():
        print(f'Loss {k:8s}: {v.item():.5f}')

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal parameters: {total_params:,}')
