from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class GenerativeMethod(nn.Module, ABC):
    @abstractmethod
    def training_loss(self, x: Tensor) -> Tensor:
        """Scalar loss for a batch in [-1, 1]."""

    @abstractmethod
    def sample(
        self,
        n: int,
        *,
        device: torch.device,
        steps: int | None = None,
    ) -> Tensor:
        """Return `n` samples in [-1, 1]; `steps` is the NFE budget."""
