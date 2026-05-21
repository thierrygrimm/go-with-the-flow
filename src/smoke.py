"""End-to-end smoke test: CIFAR-10 + mock method + FID."""
import torch

from config import SmokeConfig
from data.cifar10 import CIFAR10
from evaluate import evaluate
from methods.mock import RandomNoiseMethod
from metrics.fid import FIDMetric
from train import train


def main() -> None:
    cfg = SmokeConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = CIFAR10()
    method = RandomNoiseMethod(shape=data.shape)

    train(
        method,
        data.train_loader(cfg.train.batch_size),
        n_steps=cfg.train.n_steps,
        lr=cfg.train.lr,
        log_every=cfg.train.log_every,
        device=device,
    )

    results = evaluate(
        method,
        data.eval_loader(cfg.eval.sample_batch),
        metrics=[FIDMetric(device=device)],
        n_samples=cfg.eval.n_samples,
        sample_batch=cfg.eval.sample_batch,
        device=device,
    )

    print("\nresults:")
    for name, value in results.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
