"""Conditional flow matching (Lipman et al. 2023) with linear interpolant.

Convention: t=0 noise, t=1 data; target velocity x_1 - x_0; Euler ODE sampler.
"""
import torch
from torch import Tensor, nn

from .base import GenerativeMethod
from .unet import UNet


class FlowMatching(GenerativeMethod):
    def __init__(
        self,
        shape: tuple[int, int, int] = (3, 32, 32),
        model: nn.Module | None = None,
    ):
        super().__init__()
        self.shape = shape
        c = shape[0]
        self.model = model if model is not None else UNet(in_channels=c, out_channels=c)

    def training_loss(self, x: Tensor) -> Tensor:
        x_0 = torch.randn_like(x)
        t = torch.rand(x.shape[0], device=x.device)
        t_exp = t[:, None, None, None]
        x_t = (1 - t_exp) * x_0 + t_exp * x
        v = self.model(x_t, t)
        return ((v - (x - x_0)) ** 2).mean()

    @torch.no_grad()
    def sample(
        self,
        n: int,
        *,
        device: torch.device,
        steps: int | None = None,
    ) -> Tensor:
        steps = steps or 50
        x = torch.randn(n, *self.shape, device=device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((n,), i * dt, device=device)
            v = self.model(x, t)
            x = x + dt * v
        return x
