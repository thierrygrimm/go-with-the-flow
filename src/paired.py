"""Same-noise paired sampling: do DDPM and FM learn the same noise->image map?

Both deterministic samplers (DDPM DDIM eta=0, FM Euler) are deterministic functions
of the initial Gaussian noise. We feed the *same* noise to both and compare outputs
visually (a paired grid) and quantitatively (per-image pixel correlation). As a
control we also correlate *shuffled* pairs: if matched pairs are no more similar than
shuffled ones, the two methods are learning essentially unrelated transports.
"""
import argparse
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import torch

from data.cifar10 import CIFAR10
from ema import EMA
from methods import SIZES, build

_SAMPLER = {"ddpm": "ddim", "fm": "euler"}  # deterministic per method


def _load(name: str, size: str, results: Path, shape, device, t_embed: str):
    ckpt_path = results / f"{name}_{size}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    method = build(name, size, shape, t_embed=t_embed).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    method.load_state_dict(ckpt["model"])
    ema = None
    if "ema" in ckpt:
        ema = EMA(method).to(device)
        ema.load_state_dict(ckpt["ema"])
    method.eval()
    return method, ema


@torch.no_grad()
def _sample(method, ema, noise, *, sampler, steps, device):
    ctx = ema.swap_in(method) if ema is not None else nullcontext()
    with ctx:
        return method.sample(noise.shape[0], device=device, sampler=sampler,
                             steps=steps, x0=noise.clone())


def _corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-image Pearson correlation over flattened pixels."""
    af = a.flatten(1) - a.flatten(1).mean(1, keepdim=True)
    bf = b.flatten(1) - b.flatten(1).mean(1, keepdim=True)
    return (af * bf).sum(1) / (af.norm(dim=1) * bf.norm(dim=1) + 1e-8)


def _to_img(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()


def _to_noise(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.quantile(0.01), x.quantile(0.99)
    return ((x - lo) / (hi - lo + 1e-8)).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def plot(noise, ddpm, fm, corr, *, n_show: int, out: Path) -> None:
    n = min(n_show, ddpm.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(3 * 1.3, n * 1.3), squeeze=False)
    axes[0, 0].set_title("noise", fontsize=10)
    axes[0, 1].set_title("DDPM", fontsize=10)
    axes[0, 2].set_title("FM", fontsize=10)
    for i in range(n):
        axes[i, 0].imshow(_to_noise(noise[i]), interpolation="nearest")
        axes[i, 1].imshow(_to_img(ddpm[i]), interpolation="nearest")
        axes[i, 2].imshow(_to_img(fm[i]), interpolation="nearest")
        axes[i, 1].set_ylabel(f"r={corr[i]:.2f}", rotation=0, ha="right", va="center", fontsize=8)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Same noise, both methods", fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("./results"))
    parser.add_argument("--size", default="paper", choices=list(SIZES))
    parser.add_argument("--n-samples", type=int, default=64, help="for the correlation stats")
    parser.add_argument("--n-show", type=int, default=8, help="rows in the grid")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t-embed", default="sinusoidal", choices=["sinusoidal", "fourier"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.results / f"paired_{args.size}.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = CIFAR10().shape
    ddpm, ddpm_ema = _load("ddpm", args.size, args.results, shape, device, args.t_embed)
    fm, fm_ema = _load("fm", args.size, args.results, shape, device, args.t_embed)

    g = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(args.n_samples, *shape, generator=g, device=device)
    print(f"paired sampling (steps={args.steps}, n={args.n_samples})")
    ddpm_imgs = _sample(ddpm, ddpm_ema, noise, sampler=_SAMPLER["ddpm"], steps=args.steps, device=device)
    fm_imgs = _sample(fm, fm_ema, noise, sampler=_SAMPLER["fm"], steps=args.steps, device=device)

    matched = _corr(ddpm_imgs, fm_imgs)
    perm = torch.randperm(args.n_samples, generator=torch.Generator().manual_seed(args.seed))
    shuffled = _corr(ddpm_imgs, fm_imgs[perm])
    print(f"  matched-pair correlation : {matched.mean():.3f} +/- {matched.std():.3f}")
    print(f"  shuffled-pair correlation: {shuffled.mean():.3f} +/- {shuffled.std():.3f}")

    plt.style.use(["science", "no-latex"])
    plot(noise.cpu(), ddpm_imgs.cpu(), fm_imgs.cpu(), matched.cpu(),
         n_show=args.n_show, out=out)


if __name__ == "__main__":
    main()
