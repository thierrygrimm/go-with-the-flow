"""Plot training loss and eval FID over training steps, pulled from W&B."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import wandb

# Same Okabe-Ito palette as plot.py so the two figures read as a set.
_STYLE = {
    "ddpm": ("#0072B2", "DDPM"),   # Okabe-Ito blue
    "fm":   ("#D55E00", "FM"),     # Okabe-Ito vermilion
}


def _series(run, key) -> tuple[list[int], list[float]]:
    """Step-indexed history for `key`, deduped (last write wins) and sorted."""
    pts: dict[int, float] = {}
    for r in run.scan_history(keys=["_step", key]):
        if r.get(key) is not None:
            pts[r["_step"]] = r[key]
    xs = sorted(pts)
    return xs, [pts[x] for x in xs]


def _ema(ys: list[float], alpha: float = 0.1) -> list[float]:
    out, m = [], ys[0]
    for y in ys:
        m = alpha * y + (1 - alpha) * m
        out.append(m)
    return out


def _plot(api, path: str, size: str, seed: int, out: Path) -> None:
    fig, (ax_loss, ax_fid) = plt.subplots(1, 2, figsize=(10, 4))
    for method, (color, label) in _STYLE.items():
        name = f"{method}_{size}_seed{seed}"
        runs = api.runs(path, filters={"display_name": name})
        if not runs:
            print(f"missing run: {name}")
            continue
        run = runs[0]
        lx, ly = _series(run, "loss")
        ax_loss.plot(lx, ly, color=color, alpha=0.2, linewidth=0.7)
        ax_loss.plot(lx, _ema(ly), color=color, label=label)
        fx, fy = _series(run, "eval_fid")
        ax_fid.plot(fx, fy, marker="o", markersize=3, color=color, label=label)

    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("training loss")
    ax_loss.set_title("Training loss")
    ax_fid.set_xlabel("step")
    ax_fid.set_ylabel("eval FID")
    ax_fid.set_title("Eval FID")
    for ax in (ax_loss, ax_fid):
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("DDPM vs Flow Matching on CIFAR-10")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="fabiangroeger")
    parser.add_argument("--project", default="go-with-the-flow")
    parser.add_argument("--size", default="paper")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", type=Path, default=Path("./results"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.results / f"loss_fid_vs_step_{args.size}.png"

    plt.style.use(["science", "no-latex"])
    _plot(wandb.Api(), f"{args.entity}/{args.project}", args.size, args.seed, out)


if __name__ == "__main__":
    main()
