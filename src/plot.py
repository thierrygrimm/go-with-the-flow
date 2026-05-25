"""Plot FID-vs-NFE comparing DDPM and FM from `sweep.jsonl` files."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


_STYLE = {
    "ddpm": ("tab:blue", "DDPM"),
    "fm": ("tab:orange", "FM"),
}


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("./results"))
    parser.add_argument("--size", default="paper")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or args.results / f"fid_vs_nfe_{args.size}.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, (color, label) in _STYLE.items():
        path = args.results / f"{method}_{args.size}" / "sweep.jsonl"
        if not path.exists():
            print(f"missing: {path}")
            continue
        entries = _load(path)
        by_sampler: dict[str, list[tuple[int | None, float]]] = {}
        for e in entries:
            by_sampler.setdefault(e["sampler"], []).append((e["nfe"], e["fid"]))
        for sampler, points in by_sampler.items():
            fixed = sorted((s, f) for s, f in points if s is not None)
            leg = f"{label} ({sampler})"
            if fixed:
                xs, ys = zip(*fixed)
                ax.plot(xs, ys, marker="o", color=color, label=leg)
            for s, f in points:
                if s is None:
                    ax.axhline(f, color=color, linestyle="--", alpha=0.5, label=leg)

    ax.set_xscale("log")
    ax.set_xlabel("NFE")
    ax.set_ylabel("FID")
    ax.set_title(f"FID vs NFE on CIFAR-10 ({args.size})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
