from contextlib import contextmanager

import torch
from torch import nn


class EMA:
    """Exponential moving average of model parameters.

    Decay rule matches Ho et al.'s `tf.train.ExponentialMovingAverage(num_updates=step)`:
    `effective_decay = min(decay, (1 + step) / (10 + step))`.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.dtype.is_floating_point
        }

    def to(self, device: torch.device | str) -> "EMA":
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self

    @torch.no_grad()
    def update(self, model: nn.Module, step: int | None = None) -> None:
        decay = self.decay
        if step is not None:
            decay = min(self.decay, (1 + step) / (10 + step))
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(p.detach(), alpha=1 - decay)

    @contextmanager
    def swap_in(self, model: nn.Module):
        params = dict(model.named_parameters())
        backup = {name: params[name].data.detach().clone() for name in self.shadow}
        for name, v in self.shadow.items():
            params[name].data.copy_(v)
        try:
            yield
        finally:
            for name, v in backup.items():
                params[name].data.copy_(v)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        missing = set(self.shadow) - set(sd)
        extra = set(sd) - set(self.shadow)
        if missing or extra:
            raise ValueError(f"EMA key mismatch: missing={missing}, extra={extra}")
        self.shadow = {k: v.clone() for k, v in sd.items()}
