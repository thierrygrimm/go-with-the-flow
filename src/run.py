"""Real training entry: paper-scale defaults, EMA + checkpointing + periodic FID."""
import argparse
from pathlib import Path

import torch
import wandb

from data.cifar10 import CIFAR10
from ema import EMA
from evaluate import evaluate
from methods import METHODS, SIZES, build
from metrics.fid import FIDMetric
from train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(METHODS), default="ddpm")
    parser.add_argument("--size", choices=list(SIZES), default="paper")
    parser.add_argument("--n-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-project", default="go-with-the-flow")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(f"./results/{args.method}_{args.size}")
    resume_from = ckpt_dir / "latest.pt" if args.resume else None

    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.method}_{args.size}_seed{args.seed}",
            config=vars(args),
        )

    data = CIFAR10()
    method = build(args.method, args.size, data.shape)
    ema = EMA(method, decay=0.9999)

    n_params = sum(p.numel() for p in method.parameters()) / 1e6
    print(f"training {args.method} ({args.size}, {n_params:.1f}M params) on {device}")
    print(f"results -> {ckpt_dir}")

    fid_metric = FIDMetric(device=device)
    eval_loader = data.eval_loader(args.batch_size)

    def eval_fn(m: torch.nn.Module) -> float:
        results = evaluate(
            m, eval_loader, metrics=[fid_metric],
            n_samples=args.eval_samples,
            sample_batch=args.batch_size,
            device=device,
        )
        return results["FIDMetric"]

    train(
        method,
        data.train_loader(args.batch_size),
        n_steps=args.n_steps,
        lr=args.lr,
        log_every=100,
        device=device,
        ema=ema,
        grad_clip=1.0,
        warmup_steps=5000,
        eval_every=args.eval_every,
        eval_fn=eval_fn,
        ckpt_dir=ckpt_dir,
        resume_from=resume_from,
    )


if __name__ == "__main__":
    main()
