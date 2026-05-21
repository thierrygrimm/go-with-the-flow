"""DDPM (Ho et al. 2020): noise-prediction with linear schedule, DDIM sampler."""
import torch
from torch import Tensor, nn

from .base import GenerativeMethod
from .unet import UNet


class DDPM(GenerativeMethod):
    def __init__(
        self,
        shape: tuple[int, int, int] = (3, 32, 32),
        n_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        model: nn.Module | None = None,
    ):
        super().__init__()
        self.shape = shape
        self.n_timesteps = n_timesteps
        c = shape[0]
        self.model = model if model is not None else UNet(in_channels=c, out_channels=c)
        betas = torch.linspace(beta_start, beta_end, n_timesteps)
        self.register_buffer("alpha_bars", torch.cumprod(1.0 - betas, dim=0))

    def training_loss(self, x: Tensor) -> Tensor:
        t = torch.randint(0, self.n_timesteps, (x.shape[0],), device=x.device)
        noise = torch.randn_like(x)
        ab = self.alpha_bars[t][:, None, None, None]
        x_t = ab.sqrt() * x + (1 - ab).sqrt() * noise
        eps = self.model(x_t, t.float() / self.n_timesteps)
        return ((eps - noise) ** 2).mean()

    @torch.no_grad()
    def sample(
        self,
        n: int,
        *,
        device: torch.device,
        steps: int | None = None,
    ) -> Tensor:
        steps = steps or self.n_timesteps
        x = torch.randn(n, *self.shape, device=device)
        ts = torch.linspace(self.n_timesteps - 1, 0, steps, device=device).long()
        ab_seq = self.alpha_bars[ts]
        t_norms = ts.float() / self.n_timesteps
        for i in range(steps):
            eps = self.model(x, t_norms[i].expand(n))
            ab_cur = ab_seq[i]
            ab_next = ab_seq[i + 1] if i + 1 < steps else x.new_tensor(1.0)
            x0 = (x - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
            x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
        return x
