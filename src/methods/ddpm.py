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
        x0: Tensor | None = None,
    ) -> Tensor:
        # x0 lets callers fix the initial noise (e.g. same-noise paired sampling).
        if sampler == "ancestral":
            if steps is not None:
                raise ValueError("ancestral sampler uses n_timesteps; do not pass `steps`")
            return self._ancestral(n, device, x0=x0)
        if sampler == "heun":
            return self._heun(n, device, steps=steps or self.n_timesteps, x0=x0)
        if sampler != "ddim":
            raise ValueError(f"unknown sampler: {sampler!r}")
        steps = steps or self.n_timesteps
        x = x0 if x0 is not None else torch.randn(n, *self.shape, device=device)
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
    def _ancestral(self, n: int, device: torch.device, *, x0: Tensor | None = None) -> Tensor:
        x = x0 if x0 is not None else torch.randn(n, *self.shape, device=device)
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

    @torch.no_grad()
    def _heun(self, n: int, device: torch.device, *, steps: int, x0: Tensor | None = None) -> Tensor:
        """2nd-order Heun on the probability-flow ODE in EDM sigma-space.

        DDIM (eta=0) is exactly Euler on dx~/dsigma = eps_theta(x_t, t), with the
        change of variables x~ = x_t / sqrt(alpha_bar) and sigma = sqrt((1-ab)/ab).
        This adds the trapezoidal corrector. The corrector is skipped on the final
        step to sigma=0 (the clean image has no timestep embedding), so the cost is
        NFE = 2*steps - 1 -- the provably-2nd-order counterpart of our DDIM.
        """
        x = x0 if x0 is not None else torch.randn(n, *self.shape, device=device)
        ts = torch.linspace(self.n_timesteps - 1, 0, steps, device=device).long()
        ab = self.alpha_bars[ts]
        sigma = ((1 - ab) / ab).sqrt()
        xt = x / ab[0].sqrt()  # x~ at the first (noisiest) sigma
        for i in range(steps):
            eps1 = self.model(xt * ab[i].sqrt(), ts[i].expand(n))
            s_next = sigma[i + 1] if i + 1 < steps else sigma.new_tensor(0.0)
            dsig = s_next - sigma[i]
            xt_euler = xt + dsig * eps1
            if i + 1 < steps:
                eps2 = self.model(xt_euler * ab[i + 1].sqrt(), ts[i + 1].expand(n))
                xt = xt + 0.5 * dsig * (eps1 + eps2)
            else:
                xt = xt_euler  # sigma=0: x~ is the clean image
        self.last_nfe = 2 * steps - 1
        return xt
