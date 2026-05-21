"""FID via torchmetrics. Expects images in [-1, 1]; converts to uint8 inside."""
import torch
from torch import Tensor
from torchmetrics.image.fid import FrechetInceptionDistance

from .base import Metric


class FIDMetric(Metric):
    def __init__(self, device: str | torch.device = "cpu"):
        self._device = torch.device(device)
        self._fid = FrechetInceptionDistance(normalize=False).to(self._device)

    @staticmethod
    def _to_uint8(images: Tensor) -> Tensor:
        images = (images.clamp(-1, 1) + 1) / 2
        return (images * 255).to(torch.uint8)

    def update(self, images: Tensor, *, real: bool) -> None:
        self._fid.update(self._to_uint8(images.to(self._device)), real=real)

    def compute(self) -> float:
        return float(self._fid.compute())

    def reset(self) -> None:
        self._fid.reset()
