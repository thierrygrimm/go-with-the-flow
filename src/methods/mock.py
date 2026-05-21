import torch
from torch import Tensor, nn

from .base import GenerativeMethod


class RandomNoiseMethod(GenerativeMethod):
    def __init__(self, shape: tuple[int, int, int]):
        super().__init__()
        self.shape = shape
        self.mean = nn.Parameter(torch.zeros(1))

    def training_loss(self, x: Tensor) -> Tensor:
        return (self.mean - x.mean()) ** 2

    def sample(
        self,
        n: int,
        *,
        device: torch.device,
        steps: int | None = None,
    ) -> Tensor:
        return torch.randn(n, *self.shape, device=device)
