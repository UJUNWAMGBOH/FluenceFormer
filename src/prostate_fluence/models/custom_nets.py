import torch
import torch.nn as nn


def _gn(c: int) -> nn.GroupNorm:
    g = 8
    while c % g != 0 and g > 1:
        g //= 2
    return nn.GroupNorm(g, c)


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# =========================================================
# nnFormer 2D (minimal)
# =========================================================
class MlpConv(nn.Module):
    def __init__(self, c: int, expand: int = 4):
        super().__init__()
        h = c * expand
        self.net = nn.Sequential(
            nn.Conv2d(c, h, 1),
            _gn(h),
            nn.GELU(),
            nn.Conv2d(h, h, 3, 1, 1, groups=h),
            _gn(h),
            nn.GELU(),
            nn.Conv2d(h, c, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PoolMHSA2D(nn.Module):
    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.qkv = nn.Linear(c, c * 3)
        self.proj = nn.Linear(c, c)
        self.mha = nn.MultiheadAttention(c, heads, batch_first=True)
        self.norm = nn.LayerNorm(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)  # B, N, C
        t = self.norm(t)
        q, k, v = self.qkv(t).chunk(3, dim=-1)
        out, _ = self.mha(q, k, v, need_weights=False)
        out = self.proj(out).transpose(1, 2).reshape(B, C, H, W)
        return out


class NNFormer2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, base_c: int = 48):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_c, 3, 1, 1), _gn(base_c), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv2d(base_c, base_c * 2, 3, 2, 1), _gn(base_c * 2), nn.GELU())
        self.bott = nn.Sequential(nn.Conv2d(base_c * 2, base_c * 4, 3, 2, 1), _gn(base_c * 4), nn.GELU())

        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, 2, 2)
        self.dec2 = nn.Sequential(nn.Conv2d(base_c * 4, base_c * 2, 3, 1, 1), _gn(base_c * 2), nn.GELU())

        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, 2, 2)
        self.dec1 = nn.Sequential(nn.Conv2d(base_c * 2, base_c, 3, 1, 1), _gn(base_c), nn.GELU())

        self.out = nn.Conv2d(base_c, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        b = self.bott(x2)
        d2 = self.dec2(torch.cat([self.up2(b), x2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], 1))
        return self.out(d1)


# =========================================================
# MedFormer 2D (minimal UNet-style)
# =========================================================
class MedFormerUNet2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_ch: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool = nn.MaxPool2d(2, 2)

        self.bott = ConvBlock(base_ch * 2, base_ch * 4)

        self.up = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, 2)
        self.dec = ConvBlock(base_ch * 4, base_ch * 2)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, 2)
        self.dec2 = ConvBlock(base_ch * 2, base_ch)

        self.final = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        b = self.bott(self.pool(x2))
        d = self.dec(torch.cat([self.up(b), x2], 1))
        d2 = self.dec2(torch.cat([self.up2(d), x1], 1))
        return self.final(d2)