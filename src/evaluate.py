from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader

from methods.base import GenerativeMethod
from metrics.base import Metric


def evaluate(
    method: GenerativeMethod,
    eval_loader: DataLoader,
    metrics: Iterable[Metric],
    *,
    n_samples: int,
    sample_batch: int = 64,
    device: torch.device = torch.device("cpu"),
    sample_kwargs: dict | None = None,
) -> dict[str, float]:
    sample_kwargs = sample_kwargs or {}
    was_training = method.training
    method.to(device).eval()
    try:
        metrics = list(metrics)
        for m in metrics:
            m.reset()

        seen_real = 0
        for x, _ in eval_loader:
            x = x.to(device, non_blocking=True)
            for m in metrics:
                m.update(x, real=True)
            seen_real += x.shape[0]
            if seen_real >= n_samples:
                break

        with torch.no_grad():
            generated = 0
            while generated < n_samples:
                n = min(sample_batch, n_samples - generated)
                fake = method.sample(n, device=device, **sample_kwargs)
                for m in metrics:
                    m.update(fake, real=False)
                generated += n

        return {type(m).__name__: m.compute() for m in metrics}
    finally:
        if was_training:
            method.train()
