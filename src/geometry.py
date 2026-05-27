"""Sampling-trajectory geometry: how straight is each method's noise->image path?

Both deterministic samplers (DDPM DDIM eta=0, FM Euler) trace a polyline from the
initial noise to the final image in pixel space. We integrate each on a fixed fine
grid and measure, averaged over many seeds:

  1. arc/chord ratio   rho = sum_i ||x_{i+1}-x_i|| / ||x_N - x_0||   (>=1, =1 iff straight)
  2. speed profile      ||x_{i+1}-x_i|| as a function of progress (noise=0 -> data=1)

This is the quantitative version of the "curved vs nearly straight" motivation:
a straighter path (rho near 1, flat speed) is exactly why a coarse Euler/DDIM
discretisation costs less FID. Note rho is NOT trivially 1 for FM -- only the
*conditional* training paths are straight lines; the learned *marginal* sampling
trajectory bends, so rho is a genuine empirical measurement.
"""
import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import torch

from data.cifar10 import CIFAR10
from ema import EMA
from methods import SIZES, build
from methods.ddpm import DDPM

# Same Okabe-Ito palette as plot.py so the figures read as a set.
_STYLE = {
    "ddpm": ("#0072B2", "DDPM (DDIM)"),
    "fm":   ("#D55E00", "FM (Euler)"),
}


@torch.no_grad()
def _ddim_walk(method: DDPM, x: torch.Tensor, *, steps: int):
    """DDIM (eta=0) from noise x. Returns (arc[B], chord[B], speed_sum[steps])."""
    n = x.shape[0]
    ts = torch.linspace(method.n_timesteps - 1, 0, steps, device=x.device).long()
    ab = method.alpha_bars[ts]
    arc = torch.zeros(n, device=x.device)
    speed = torch.zeros(steps, device=x.device)
    cur, x_init = x, x
    for i in range(steps):
        eps = method.model(cur, ts[i].expand(n))
        ab_next = ab[i + 1] if i + 1 < steps else cur.new_tensor(1.0)
        x0_pred = (cur - (1 - ab[i]).sqrt() * eps) / ab[i].sqrt()
        nxt = ab_next.sqrt() * x0_pred + (1 - ab_next).sqrt() * eps
        d = (nxt - cur).flatten(1).norm(dim=1)
        arc += d
        speed[i] = d.sum()
        cur = nxt
    chord = (cur - x_init).flatten(1).norm(dim=1)
    return arc, chord, speed


@torch.no_grad()
def _euler_walk(method, x: torch.Tensor, *, steps: int):
    """FM Euler (t: 0->1) from noise x. Returns (arc[B], chord[B], speed_sum[steps])."""
    n = x.shape[0]
    dt = 1.0 / steps
    arc = torch.zeros(n, device=x.device)
    speed = torch.zeros(steps, device=x.device)
    cur, x_init = x, x
    for i in range(steps):
        t = torch.full((n,), i * dt, device=x.device)
        nxt = cur + dt * method.model(cur, t * 1000.0)
        d = (nxt - cur).flatten(1).norm(dim=1)
        arc += d
        speed[i] = d.sum()
        cur = nxt
    chord = (cur - x_init).flatten(1).norm(dim=1)
    return arc, chord, speed


def measure(method, walk, *, n_samples: int, batch: int, steps: int, device, seed: int) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)
    arcs, chords, speed_sum, done = [], [], torch.zeros(steps, device=device), 0
    while done < n_samples:
        b = min(batch, n_samples - done)
        x = torch.randn(b, *method.shape, generator=g, device=device)
        arc, chord, speed = walk(method, x, steps=steps)
        arcs.append(arc.cpu())
        chords.append(chord.cpu())
        speed_sum += speed
        done += b
    rho = torch.cat(arcs) / torch.cat(chords)
    print(f"  rho = {rho.mean():.3f} +/- {rho.std():.3f}  (n={done}, steps={steps})")
    return {
        "rho_mean": rho.mean().item(),
        "rho_std": rho.std().item(),
        "speed": (speed_sum / done).cpu().tolist(),
        "steps": steps,
        "n_samples": done,
    }


def plot(results: dict, out: Path) -> None:
    fig, (ax_speed, ax_rho) = plt.subplots(1, 2, figsize=(10, 4))
    for method, (color, label) in _STYLE.items():
        if method not in results:
            continue
        speed = results[method]["speed"]
        xs = [i / (len(speed) - 1) for i in range(len(speed))]
        ax_speed.plot(xs, speed, color=color, label=label)
    ax_speed.set_xlabel("sampling progress (noise $\\to$ data)")
    ax_speed.set_ylabel(r"step length $\|x_{i+1}-x_i\|$")
    ax_speed.set_title("Where each method moves")
    ax_speed.legend()
    ax_speed.grid(True, alpha=0.3)

    # Plot excess length (rho - 1) so the y=0 baseline IS the straight line and the
    # ~3x gap is visible; bars of rho itself (~1.0) look identical and bury the result.
    methods = [m for m in _STYLE if m in results]
    excess = [results[m]["rho_mean"] - 1.0 for m in methods]
    err = [results[m]["rho_std"] for m in methods]
    ax_rho.bar(range(len(methods)), excess, yerr=err, capsize=4,
               color=[_STYLE[m][0] for m in methods])
    for x, m in enumerate(methods):
        ax_rho.annotate(rf"$\rho={results[m]['rho_mean']:.3f}$",
                        (x, excess[x] + err[x]), textcoords="offset points",
                        xytext=(0, 4), ha="center", va="bottom", fontsize=9)
    # headroom above the tallest error bar so the rho annotation doesn't clip the top spine
    ax_rho.set_ylim(0, max(e + r for e, r in zip(excess, err)) * 1.25)
    ax_rho.set_xticks(range(len(methods)))
    ax_rho.set_xticklabels([_STYLE[m][1] for m in methods])
    ax_rho.set_ylabel(r"excess path length  (arc/chord $-$ 1)")
    ax_rho.set_title("Path straightness (0 = straight line)")
    ax_rho.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Sampling-trajectory geometry on CIFAR-10")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("./results"))
    parser.add_argument("--size", default="paper", choices=list(SIZES))
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--steps", type=int, default=250, help="fixed fine grid for both methods")
    parser.add_argument("--sample-batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t-embed", default="sinusoidal", choices=["sinusoidal", "fourier"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.results / f"geometry_{args.size}.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = CIFAR10()
    results: dict[str, dict] = {}
    for method_name in _STYLE:
        ckpt_path = args.results / f"{method_name}_{args.size}" / "best.pt"
        if not ckpt_path.exists():
            print(f"missing: {ckpt_path}")
            continue
        method = build(method_name, args.size, data.shape, t_embed=args.t_embed).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        method.load_state_dict(ckpt["model"])
        ema = None
        if "ema" in ckpt:
            ema = EMA(method).to(device)
            ema.load_state_dict(ckpt["ema"])
        method.eval()
        walk = _ddim_walk if isinstance(method, DDPM) else _euler_walk
        print(f"geometry {method_name} ({args.size}) from {ckpt_path}")
        ctx = ema.swap_in(method) if ema is not None else nullcontext()
        with ctx:
            results[method_name] = measure(
                method, walk, n_samples=args.n_samples, batch=args.sample_batch,
                steps=args.steps, device=device, seed=args.seed,
            )

    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps(results, indent=2))
    print(f"saved: {json_out}")
    if results:
        plt.style.use(["science", "no-latex"])
        plot(results, out)


if __name__ == "__main__":
    main()
