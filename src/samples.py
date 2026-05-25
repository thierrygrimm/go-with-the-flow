"""Sample grid PNG from a trained checkpoint."""
import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torchvision.utils as vutils

from data.cifar10 import CIFAR10
from ema import EMA
from methods import METHODS, SIZES, build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(METHODS), default="ddpm")
    parser.add_argument("--size", choices=list(SIZES), default="paper")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--sampler", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t-embed", default="sinusoidal", choices=["sinusoidal", "fourier"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt or Path(f"./results/{args.method}_{args.size}/best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    out_path = args.out or ckpt_path.parent / "samples.png"

    data = CIFAR10()
    method = build(args.method, args.size, data.shape, t_embed=args.t_embed).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    method.load_state_dict(ckpt["model"])

    ema: EMA | None = None
    if "ema" in ckpt:
        ema = EMA(method).to(device)
        ema.load_state_dict(ckpt["ema"])

    sampler = args.sampler or type(method).paper_sampler
    sample_kwargs: dict = {"sampler": sampler}
    if args.steps is not None:
        sample_kwargs["steps"] = args.steps

    method.eval()
    ctx = ema.swap_in(method) if ema is not None else nullcontext()
    with ctx, torch.no_grad():
        samples = method.sample(args.n, device=device, **sample_kwargs)

    samples = (samples.clamp(-1, 1) + 1) / 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(samples, out_path, nrow=args.nrow)
    print(f"saved: {out_path}  ({args.n} samples, sampler={sampler})")


if __name__ == "__main__":
    main()
