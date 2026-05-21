"""Final paper-FID evaluation: load a trained checkpoint, sample with the paper sampler."""
import argparse
from contextlib import nullcontext
from pathlib import Path

import torch

from data.cifar10 import CIFAR10
from ema import EMA
from evaluate import evaluate
from methods import METHODS, SIZES, build
from metrics.fid import FIDMetric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(METHODS), default="ddpm")
    parser.add_argument("--size", choices=list(SIZES), default="paper")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="checkpoint path (defaults to results/<method>_<size>/best.pt)")
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument("--sample-batch", type=int, default=128)
    parser.add_argument("--sampler", default=None,
                        help="override paper-default sampler (ddpm: ancestral, fm: rk45)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt or Path(f"./results/{args.method}_{args.size}/best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    data = CIFAR10()
    method = build(args.method, args.size, data.shape).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    method.load_state_dict(ckpt["model"])

    ema: EMA | None = None
    if "ema" in ckpt:
        ema = EMA(method).to(device)
        ema.load_state_dict(ckpt["ema"])

    sampler = args.sampler or type(method).paper_sampler
    print(f"eval {args.method} ({args.size}, sampler={sampler}, n={args.n_samples}) from {ckpt_path}")

    fid_metric = FIDMetric(device=device)
    eval_loader = data.eval_loader(args.sample_batch)

    ctx = ema.swap_in(method) if ema is not None else nullcontext()
    with ctx:
        results = evaluate(
            method, eval_loader, [fid_metric],
            n_samples=args.n_samples,
            sample_batch=args.sample_batch,
            device=device,
            sample_kwargs={"sampler": sampler},
        )

    print(f"FID: {results['FIDMetric']:.4f}")


if __name__ == "__main__":
    main()
