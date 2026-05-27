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

    def compute_components(self) -> dict[str, float]:
        """Split FID = ||mu_r - mu_f||^2 + Tr(S_r + S_f - 2 sqrt(S_r S_f)).

        The first (mean) term tracks where the feature centroids sit -- roughly
        fidelity; the second (covariance) term compares feature spread -- roughly
        diversity. The mean term is computed directly from the running feature
        sums; the covariance term is the remainder of torchmetrics' own FID, so
        we never reimplement the matrix square root.
        """
        f = self._fid
        mean_real = f.real_features_sum / f.real_features_num_samples
        mean_fake = f.fake_features_sum / f.fake_features_num_samples
        diff = mean_real - mean_fake
        mean_term = float(diff.dot(diff))
        fid = float(f.compute())
        return {"fid": fid, "fid_mean_term": mean_term, "fid_cov_term": fid - mean_term}

    def reset(self) -> None:
        self._fid.reset()
