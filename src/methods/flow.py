"""Conditional flow matching (Lipman et al. 2023) with linear/OT interpolant.

t=0 is noise, t=1 is data. Target velocity: `x_1 - (1 - sigma_min) * x_0`.
Samplers: Euler (default, fixed-step) and adaptive RK45 (paper sampler).
"""
import torch
import torchdiffeq
from torch import Tensor, nn

from .base import GenerativeMethod
from .unet import UNet


# t in [0, 1] is scaled to a [0, T]-style integer-like range for the shared U-Net's embedding.
_T_SCALE = 1000.0


class FlowMatching(GenerativeMethod):
    paper_sampler = "rk45"

    def __init__(
        self,
        shape: tuple[int, int, int] = (3, 32, 32),
        sigma_min: float = 1e-4,
        model: nn.Module | None = None,
    ):
        super().__init__()
        self.shape = shape
        self.sigma_min = sigma_min
        c = shape[0]
        self.model = model if model is not None else UNet(in_channels=c, out_channels=c)

    def training_loss(self, x: Tensor) -> Tensor:
        x_0 = torch.randn_like(x)
        t = torch.rand(x.shape[0], device=x.device)
        t_exp = t[:, None, None, None]
        alpha = 1 - self.sigma_min
        x_t = (1 - alpha * t_exp) * x_0 + t_exp * x
        target = x - alpha * x_0
        v = self.model(x_t, t * _T_SCALE)
        return ((v - target) ** 2).mean()

    @torch.no_grad()
    def sample(
        self,
        n: int,
        *,
        device: torch.device,
        steps: int | None = None,
        sampler: str = "euler",
    ) -> Tensor:
        x = torch.randn(n, *self.shape, device=device)
        if sampler == "rk45":
            calls = [0]
            def f(t: Tensor, x: Tensor) -> Tensor:
                calls[0] += 1
                return self.model(x, (t * _T_SCALE).expand(x.shape[0]))
            t_span = torch.tensor([0.0, 1.0], device=device)
            traj = torchdiffeq.odeint(f, x, t_span, method="dopri5", rtol=1e-5, atol=1e-5)
            self.last_nfe = calls[0]
            return traj[-1]
        if sampler != "euler":
            raise ValueError(f"unknown sampler: {sampler!r}")
        steps = steps or 50
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((n,), i * dt, device=device)
            v = self.model(x, t * _T_SCALE)
            x = x + dt * v
        return x
