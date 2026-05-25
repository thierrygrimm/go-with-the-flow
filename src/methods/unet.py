import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


_PRESETS: dict[str, dict] = {
    "small": dict(base=64, ch_mults=(1, 2, 2, 2), attn_levels=(1,), dropout=0.0),
    "paper": dict(base=128, ch_mults=(1, 2, 2, 2), attn_levels=(1,), dropout=0.1),
}
SIZES = tuple(_PRESETS)

_GN_GROUPS = 32  # Ho et al.: tf.contrib.layers.group_norm default


def sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    """Ho et al.'s `get_timestep_embedding`. `t` is in the [0, T] integer-like range."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)
    )
    args = t.to(torch.float32)[:, None] * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class _SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        return sinusoidal_embedding(t, self.dim)


class _FourierEmbedding(nn.Module):
    """Random Fourier features (Tancik et al. 2020). Frequencies fixed at init."""

    def __init__(self, dim: int, scale: float = 16.0, t_max: float = 1000.0):
        super().__init__()
        self.register_buffer("freqs", torch.randn(dim // 2) * scale / t_max)

    def forward(self, t: Tensor) -> Tensor:
        args = t.to(torch.float32)[:, None] * self.freqs[None, :] * 2 * math.pi
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_GN_GROUPS, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(_GN_GROUPS, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
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
        q = q.reshape(b, c, h * w).transpose(1, 2)
        k = k.reshape(b, c, h * w).transpose(1, 2)
        v = v.reshape(b, c, h * w).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, scale=c ** -0.5)
        return x + self.proj(out.transpose(1, 2).reshape(b, c, h, w))


class _Stage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float, attn: bool):
        super().__init__()
        self.rb = ResBlock(in_ch, out_ch, t_dim, dropout=dropout)
        self.attn = Attention(out_ch) if attn else nn.Identity()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.attn(self.rb(x, t))


def _downsample(ch: int) -> nn.Module:
    return nn.Conv2d(ch, ch, 3, stride=2, padding=1)


def _upsample(ch: int) -> nn.Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest-exact"),
        nn.Conv2d(ch, ch, 3, padding=1),
    )


class UNet(nn.Module):
    """Ho et al. 2020 DDPM U-Net. `(x, t) -> tensor with same shape as x`."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base: int = 64,
        ch_mults: tuple[int, ...] = (1, 2, 2, 2),
        attn_levels: tuple[int, ...] = (1,),
        dropout: float = 0.0,
        num_res_blocks: int = 2,
        t_embed: str = "sinusoidal",
    ):
        super().__init__()
        t_dim = 4 * base
        if t_embed == "sinusoidal":
            self.t_embedder: nn.Module = _SinusoidalEmbedding(base)
        elif t_embed == "fourier":
            self.t_embedder = _FourierEmbedding(base)
        else:
            raise ValueError(f"unknown t_embed: {t_embed!r}")
        self.time_mlp = nn.Sequential(
            nn.Linear(base, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.init_conv = nn.Conv2d(in_channels, base, 3, padding=1)

        n_levels = len(ch_mults)
        current_ch = base
        skip_chs: list[int] = [base]

        self.downs = nn.ModuleList()
        for level in range(n_levels):
            out_ch = base * ch_mults[level]
            attn = level in attn_levels
            for _ in range(num_res_blocks):
                self.downs.append(_Stage(current_ch, out_ch, t_dim, dropout, attn))
                current_ch = out_ch
                skip_chs.append(current_ch)
            if level != n_levels - 1:
                self.downs.append(_downsample(current_ch))
                skip_chs.append(current_ch)

        self.mid1 = ResBlock(current_ch, current_ch, t_dim, dropout=dropout)
        self.mid_attn = Attention(current_ch)
        self.mid2 = ResBlock(current_ch, current_ch, t_dim, dropout=dropout)

        self.ups = nn.ModuleList()
        for level in reversed(range(n_levels)):
            out_ch = base * ch_mults[level]
            attn = level in attn_levels
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_chs.pop()
                self.ups.append(_Stage(current_ch + skip_ch, out_ch, t_dim, dropout, attn))
                current_ch = out_ch
            if level != 0:
                self.ups.append(_upsample(current_ch))

        self.final_norm = nn.GroupNorm(_GN_GROUPS, base)
        self.final_conv = nn.Conv2d(base, out_channels, 3, padding=1)

        # Ho et al. nn.py `default_init` (xavier-uniform); `init_scale=0` zeroes ResBlock.conv2,
        # Attention.proj, and the final conv_out.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, ResBlock):
                nn.init.zeros_(m.conv2.weight)
                nn.init.zeros_(m.conv2.bias)
            elif isinstance(m, Attention):
                nn.init.zeros_(m.proj.weight)
                nn.init.zeros_(m.proj.bias)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    @staticmethod
    def for_size(size: str, in_channels: int = 3, out_channels: int = 3, t_embed: str = "sinusoidal") -> "UNet":
        preset = _PRESETS.get(size)
        if preset is None:
            raise ValueError(f"unknown size: {size!r}")
        return UNet(in_channels, out_channels, t_embed=t_embed, **preset)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_emb = self.time_mlp(self.t_embedder(t))
        h = self.init_conv(x)
        skips = [h]
        for block in self.downs:
            h = block(h, t_emb) if isinstance(block, _Stage) else block(h)
            skips.append(h)
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)
        for block in self.ups:
            if isinstance(block, _Stage):
                h = block(torch.cat([h, skips.pop()], dim=1), t_emb)
            else:
                h = block(h)
        return self.final_conv(F.silu(self.final_norm(h)))
