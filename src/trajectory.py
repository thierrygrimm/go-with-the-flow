"""Trajectory visualization: noise -> image along the sampling trajectory.

For each NFE budget in `--nfe-budgets` (default 5 / 50 / 1000), runs the fast
fixed-step sampler (DDIM for DDPM, Euler for FM) on a shared batch of seeded
noise and records the state after every step. Then renders a grid:

  rows = (NFE block) x (n samples per block)
  cols = `--n-snapshots` evenly-spaced intermediates from noise (left) to final (right)

Same seed across all NFE blocks, so each column shows "the same trajectory at
different step counts": low-NFE rows are coarse and end at a worse image; the
high-NFE row is smooth and ends well.

Saves to `results/<method>_<size>/trajectory.png`.
"""
import argparse
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data.cifar10 import CIFAR10
from ema import EMA
from methods import METHODS, SIZES, build
from methods.ddpm import DDPM
from methods.flow import FlowMatching


@torch.no_grad()
def _ddim_trajectory(method: DDPM, n: int, device, *, steps: int, seed: int) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(n, *method.shape, device=device)
    snaps = [x.clone()]
    ts = torch.linspace(method.n_timesteps - 1, 0, steps, device=device).long()
    ab_seq = method.alpha_bars[ts]
    for i in range(steps):
        eps = method.model(x, ts[i].expand(n))
        ab_cur = ab_seq[i]
        ab_next = ab_seq[i + 1] if i + 1 < steps else x.new_tensor(1.0)
        x0 = (x - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
        x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
        snaps.append(x.clone())
    return snaps


@torch.no_grad()
def _fm_euler_trajectory(method: FlowMatching, n: int, device, *, steps: int, seed: int) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(n, *method.shape, device=device)
    snaps = [x.clone()]
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i * dt, device=device)
        v = method.model(x, t * 1000.0)
        x = x + dt * v
        snaps.append(x.clone())
    return snaps


def render(method, *, n_samples: int, nfe_budgets: list[int], n_snapshots: int,
           device, seed: int, out_path: Path) -> None:
    fn = _ddim_trajectory if isinstance(method, DDPM) else _fm_euler_trajectory
    sampler_name = "DDIM" if isinstance(method, DDPM) else "Euler"
    method_name = "DDPM" if isinstance(method, DDPM) else "FM"

    # Capture snapshots per NFE budget, then subsample to n_snapshots columns.
    blocks: list[tuple[int, torch.Tensor]] = []
    for steps in nfe_budgets:
        print(f"  NFE={steps} ...", flush=True)
        snaps = fn(method, n_samples, device, steps=steps, seed=seed)
        n_cols = min(n_snapshots, len(snaps))
        # Evenly spaced indices from 0 (noise) to len(snaps)-1 (final).
        idx = [round(i * (len(snaps) - 1) / (n_cols - 1)) for i in range(n_cols)]
        selected = torch.stack([snaps[i] for i in idx], dim=1).cpu()  # [n, n_cols, C, H, W]
        blocks.append((steps, selected))

    n_cols = max(b[1].shape[1] for b in blocks)
    n_rows = n_samples * len(blocks)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.0, n_rows * 1.0 + 0.4),
        gridspec_kw={"wspace": 0.03, "hspace": 0.03},
    )
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    for block_i, (steps, batch) in enumerate(blocks):
        bn_cols = batch.shape[1]
        for s in range(n_samples):
            row = block_i * n_samples + s
            for c in range(bn_cols):
                ax = axes[row, c]
                img = (batch[s, c].clamp(-1, 1) + 1) / 2
                ax.imshow(img.permute(1, 2, 0).numpy())
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            for c in range(bn_cols, n_cols):
                axes[row, c].axis("off")
            if s == 0:
                axes[row, 0].set_ylabel(
                    f"NFE={steps}", rotation=0, ha="right", va="center",
                    labelpad=22, fontsize=10,
                )

    axes[0, 0].set_title("noise", fontsize=9)
    axes[0, -1].set_title("final", fontsize=9)
    fig.suptitle(f"{method_name} {sampler_name} trajectory: noise -> image", fontsize=11, y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(METHODS), default="ddpm")
    parser.add_argument("--size", choices=list(SIZES), default="paper")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--t-embed", default="sinusoidal", choices=["sinusoidal", "fourier"])
    parser.add_argument("--n-samples", type=int, default=4, help="samples per NFE block")
    parser.add_argument("--n-snapshots", type=int, default=8, help="columns per row")
    parser.add_argument("--nfe-budgets", type=int, nargs="+", default=[5, 50, 1000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt or Path(f"./results/{args.method}_{args.size}/best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    out_path = args.out or ckpt_path.parent / "trajectory.png"

    data = CIFAR10()
    method = build(args.method, args.size, data.shape, t_embed=args.t_embed).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    method.load_state_dict(ckpt["model"])

    ema: EMA | None = None
    if "ema" in ckpt:
        ema = EMA(method).to(device)
        ema.load_state_dict(ckpt["ema"])

    method.eval()
    print(f"trajectory {args.method} ({args.size}) NFE={args.nfe_budgets} -> {out_path}")
    ctx = ema.swap_in(method) if ema is not None else nullcontext()
    with ctx:
        render(method, n_samples=args.n_samples, nfe_budgets=args.nfe_budgets,
               n_snapshots=args.n_snapshots, device=device, seed=args.seed,
               out_path=out_path)


if __name__ == "__main__":
    main()
