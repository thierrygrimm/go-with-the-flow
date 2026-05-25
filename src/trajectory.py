"""Trajectory visualization: same noise, different NFE budgets.

DDIM (eta=0) and FM Euler are both fully deterministic given the initial noise,
so we draw N initial noises once and run each NFE budget against all of them.
The figure is one panel per noise; inside a panel, rows are NFE budgets and
columns are evenly-spaced snapshots from noise (left) to final image (right).

Reading the figure: within a panel, every row starts from the same noise. The
low-NFE rows are a coarse Euler/DDIM discretisation of the same trajectory the
high-NFE row resolves accurately, so the final image converges as NFE grows.
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
def _ddim_trajectory(method: DDPM, x: torch.Tensor, *, steps: int) -> list[torch.Tensor]:
    """DDIM (eta=0) starting from `x`. Returns initial state + state after each step."""
    n = x.shape[0]
    snaps = [x.clone()]
    ts = torch.linspace(method.n_timesteps - 1, 0, steps, device=x.device).long()
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
def _fm_euler_trajectory(method: FlowMatching, x: torch.Tensor, *, steps: int) -> list[torch.Tensor]:
    """Euler integration of dx/dt = v_theta(x, t) from t=0 (noise) to t=1 (data)."""
    n = x.shape[0]
    snaps = [x.clone()]
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i * dt, device=x.device)
        v = method.model(x, t * 1000.0)
        x = x + dt * v
        snaps.append(x.clone())
    return snaps


def _to_display(x: torch.Tensor) -> torch.Tensor:
    """Map a CHW tensor with arbitrary range to [0, 1] via 1st-99th percentile.

    For final images (values in [-1, 1]) this is near-identity; for Gaussian
    noise (~N(0, 1), 99th percentile ~2.3) it expands the range and produces a
    proper grainy noise look instead of the saturated R/G/B test-pattern you get
    by clamping to [-1, 1].
    """
    lo = x.quantile(0.01)
    hi = x.quantile(0.99)
    return ((x - lo) / (hi - lo + 1e-8)).clamp(0, 1)


def render(method, *, n_noises: int, nfe_budgets: list[int], n_snapshots: int,
           device, seed: int, out_path: Path) -> None:
    fn = _ddim_trajectory if isinstance(method, DDPM) else _fm_euler_trajectory
    sampler_name = "DDIM" if isinstance(method, DDPM) else "Euler"
    method_name = "DDPM" if isinstance(method, DDPM) else "FM"

    # One batch of seeded noises, shared across every NFE budget.
    g = torch.Generator(device=device).manual_seed(seed)
    initial = torch.randn(n_noises, *method.shape, generator=g, device=device)

    # For each NFE budget, run a single batched trajectory then subsample columns.
    per_nfe: dict[int, torch.Tensor] = {}  # nfe -> [n_noises, n_cols, C, H, W]
    for steps in nfe_budgets:
        print(f"  NFE={steps} ...", flush=True)
        snaps = fn(method, initial.clone(), steps=steps)
        n_cols_block = min(n_snapshots, len(snaps))
        idx = [round(i * (len(snaps) - 1) / (n_cols_block - 1)) for i in range(n_cols_block)]
        per_nfe[steps] = torch.stack([snaps[i] for i in idx], dim=1).cpu()

    n_cols = max(t.shape[1] for t in per_nfe.values())
    rows_per_panel = len(nfe_budgets)
    cell = 1.05

    fig = plt.figure(
        figsize=(n_cols * cell + 0.6, rows_per_panel * n_noises * cell + 1.0 + 0.25 * n_noises),
        layout="constrained",
    )
    fig.suptitle(
        f"{method_name} ({sampler_name}) - same noise, different NFE budgets",
        fontsize=13, fontweight="bold",
    )
    subfigs = fig.subfigures(n_noises, 1, hspace=0.04)
    if n_noises == 1:
        subfigs = [subfigs]

    for noise_i, sf in enumerate(subfigs):
        sf.suptitle(f"noise #{noise_i + 1}", fontsize=10, fontweight="bold",
                    ha="left", x=0.005)
        axes = sf.subplots(
            rows_per_panel, n_cols,
            gridspec_kw={"wspace": 0.04, "hspace": 0.04},
            squeeze=False,
        )
        for row_i, nfe in enumerate(nfe_budgets):
            batch = per_nfe[nfe]
            bn_cols = batch.shape[1]
            for c in range(n_cols):
                ax = axes[row_i, c]
                if c < bn_cols:
                    img = _to_display(batch[noise_i, c])
                    ax.imshow(img.permute(1, 2, 0).numpy(), interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            axes[row_i, 0].set_ylabel(
                f"NFE={nfe}", rotation=0, ha="right", va="center",
                labelpad=12, fontsize=9,
            )
        # "noise" / "final" markers above the first row of the first panel only
        if noise_i == 0:
            axes[0, 0].set_title("noise", fontsize=8, color="0.4")
            axes[0, -1].set_title("final", fontsize=8, color="0.4")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(METHODS), default="ddpm")
    parser.add_argument("--size", choices=list(SIZES), default="paper")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--t-embed", default="sinusoidal", choices=["sinusoidal", "fourier"])
    parser.add_argument("--n-noises", type=int, default=3,
                        help="number of distinct initial noises (= number of panels)")
    parser.add_argument("--n-snapshots", type=int, default=8,
                        help="snapshot columns per row (incl. start and final)")
    parser.add_argument("--nfe-budgets", type=int, nargs="+", default=[5, 50, 1000],
                        help="step counts; rows inside each panel")
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
    print(f"trajectory {args.method} ({args.size}) "
          f"noises={args.n_noises} NFE={args.nfe_budgets} -> {out_path}")
    ctx = ema.swap_in(method) if ema is not None else nullcontext()
    with ctx:
        render(method, n_noises=args.n_noises, nfe_budgets=args.nfe_budgets,
               n_snapshots=args.n_snapshots, device=device, seed=args.seed,
               out_path=out_path)


if __name__ == "__main__":
    main()
