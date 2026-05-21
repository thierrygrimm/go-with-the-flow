"""Smoke test: train a method briefly, then compute FID."""
import argparse

import torch

from config import SmokeConfig
from data.cifar10 import CIFAR10
from evaluate import evaluate
from methods import METHODS, SIZES, build
from metrics.fid import FIDMetric
from train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(METHODS), default="ddpm")
    parser.add_argument("--size", choices=list(SIZES), default="small")
    args = parser.parse_args()

    cfg = SmokeConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = CIFAR10()
    method = build(args.method, args.size, data.shape)
    n_params = sum(p.numel() for p in method.parameters()) / 1e6
    print(f"running {args.method} ({args.size}, {n_params:.1f}M params) on {device}")

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

    print(f"\n{args.method} ({args.size}):")
    for name, value in results.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
