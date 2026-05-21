import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


_GN_GROUPS = 8
_EMBED_T_SCALE = 1000.0
_PRESETS: dict[str, dict] = {
    "small": dict(base=64, ch_mults=(1, 2, 2, 2), attn_levels=(1,)),
    "paper": dict(base=128, ch_mults=(1, 2, 2, 2), attn_levels=(1,)),
}
SIZES = tuple(_PRESETS)


def sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    """Sinusoidal embedding of `t` in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t[:, None] * _EMBED_T_SCALE * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_GN_GROUPS, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(_GN_GROUPS, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(_GN_GROUPS, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)
        attn = (q @ k * c ** -0.5).softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    """U-Net taking `(x, t)` to a tensor with the same shape as x. `t` is (B,) in [0, 1]."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base: int = 64,
        ch_mults: tuple[int, ...] = (1, 2, 2, 2),
        attn_levels: tuple[int, ...] = (1,),
    ):
        super().__init__()
        t_dim = 4 * base
        self.sin_dim = base
        self.time_mlp = nn.Sequential(
            nn.Linear(base, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.init_conv = nn.Conv2d(in_channels, base, 3, padding=1)

        chs = [base * m for m in ch_mults]
        in_outs = list(zip([base] + chs[:-1], chs))
        n_levels = len(in_outs)

        self.downs = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(in_outs):
            is_last = i == n_levels - 1
            self.downs.append(nn.ModuleList([
                ResBlock(d_in, d_out, t_dim),
                ResBlock(d_out, d_out, t_dim),
                Attention(d_out) if i in attn_levels else nn.Identity(),
                nn.Conv2d(d_out, d_out, 3, stride=2, padding=1) if not is_last else nn.Identity(),
            ]))

        mid = chs[-1]
        self.mid1 = ResBlock(mid, mid, t_dim)
        self.mid_attn = Attention(mid)
        self.mid2 = ResBlock(mid, mid, t_dim)

        self.ups = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(reversed(in_outs)):
            is_last = i == n_levels - 1
            level_idx = n_levels - 1 - i
            self.ups.append(nn.ModuleList([
                ResBlock(d_out + d_out, d_out, t_dim),
                ResBlock(d_out + d_out, d_in, t_dim),
                Attention(d_in) if level_idx in attn_levels else nn.Identity(),
                nn.ConvTranspose2d(d_in, d_in, 4, stride=2, padding=1) if not is_last else nn.Identity(),
            ]))

        self.final_norm = nn.GroupNorm(_GN_GROUPS, base)
        self.final_conv = nn.Conv2d(base, out_channels, 3, padding=1)

    @staticmethod
    def for_size(size: str, in_channels: int = 3, out_channels: int = 3) -> "UNet":
        preset = _PRESETS.get(size)
        if preset is None:
            raise ValueError(f"unknown size: {size!r}")
        return UNet(in_channels, out_channels, **preset)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.sin_dim))
        h = self.init_conv(x)
        skips = []
        for rb1, rb2, attn, down in self.downs:
            h = rb1(h, t_emb)
            skips.append(h)
            h = rb2(h, t_emb)
            h = attn(h)
            skips.append(h)
            h = down(h)
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)
        for rb1, rb2, attn, up in self.ups:
            h = rb1(torch.cat([h, skips.pop()], dim=1), t_emb)
            h = rb2(torch.cat([h, skips.pop()], dim=1), t_emb)
            h = attn(h)
            h = up(h)
        return self.final_conv(F.silu(self.final_norm(h)))
