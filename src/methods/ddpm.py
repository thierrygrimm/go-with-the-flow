"""DDPM (Ho et al. 2020): noise-prediction with linear / cosine schedule, DDIM + ancestral samplers."""
import math

import torch
from torch import Tensor, nn

from .base import GenerativeMethod
from .unet import UNet


def _cosine_betas(n_timesteps: int, s: float = 0.008) -> Tensor:
    """Cosine β schedule from Nichol & Dhariwal 2021 (Improved DDPM)."""
    t = torch.linspace(0, n_timesteps, n_timesteps + 1) / n_timesteps
    ab = torch.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2
    ab = ab / ab[0]
    return (1 - ab[1:] / ab[:-1]).clamp(max=0.999)


class DDPM(GenerativeMethod):
    paper_sampler = "ancestral"

    def __init__(
        self,
        shape: tuple[int, int, int] = (3, 32, 32),
        n_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
        model: nn.Module | None = None,
    ):
        super().__init__()
        self.shape = shape
        self.n_timesteps = n_timesteps
        c = shape[0]
        self.model = model if model is not None else UNet(in_channels=c, out_channels=c)
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, n_timesteps)
        elif schedule == "cosine":
            betas = _cosine_betas(n_timesteps)
        else:
            raise ValueError(f"unknown schedule: {schedule!r}")
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", torch.cumprod(1.0 - betas, dim=0))

    def training_loss(self, x: Tensor) -> Tensor:
        t = torch.randint(0, self.n_timesteps, (x.shape[0],), device=x.device)
        noise = torch.randn_like(x)
        ab = self.alpha_bars[t][:, None, None, None]
        x_t = ab.sqrt() * x + (1 - ab).sqrt() * noise
        eps = self.model(x_t, t)
        return ((eps - noise) ** 2).mean()

    @torch.no_grad()
    def sample(
        self,
        n: int,
        *,
        device: torch.device,
        steps: int | None = None,
        sampler: str = "ddim",
    ) -> Tensor:
        if sampler == "ancestral":
            if steps is not None:
                raise ValueError("ancestral sampler uses n_timesteps; do not pass `steps`")
            return self._ancestral(n, device)
        if sampler != "ddim":
            raise ValueError(f"unknown sampler: {sampler!r}")
        steps = steps or self.n_timesteps
        x = torch.randn(n, *self.shape, device=device)
        ts = torch.linspace(self.n_timesteps - 1, 0, steps, device=device).long()
        ab_seq = self.alpha_bars[ts]
        for i in range(steps):
            eps = self.model(x, ts[i].expand(n))
            ab_cur = ab_seq[i]
            ab_next = ab_seq[i + 1] if i + 1 < steps else x.new_tensor(1.0)
            x0 = (x - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
            x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
        return x

    @torch.no_grad()
    def _ancestral(self, n: int, device: torch.device) -> Tensor:
        x = torch.randn(n, *self.shape, device=device)
        for t in reversed(range(self.n_timesteps)):
            t_batch = torch.full((n,), t, device=device, dtype=torch.long)
            eps = self.model(x, t_batch)
            beta = self.betas[t]
            alpha = 1.0 - beta
            ab = self.alpha_bars[t]
            mean = (x - beta / (1 - ab).sqrt() * eps) / alpha.sqrt()
            if t > 0:
                x = mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x
