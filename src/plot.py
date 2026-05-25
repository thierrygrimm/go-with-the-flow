"""Plot FID vs NFE comparing DDPM and FM, plus a delta panel for FM's advantage."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401


# DDIM/Euler are the fixed-step samplers we sweep NFE on. ancestral/RK45 are paper baselines.
_FIXED_STEP_SAMPLER = {"ddpm": "ddim", "fm": "euler"}
_STYLE = {
    "ddpm": ("C0", "DDPM"),
    "fm": ("C1", "FM"),
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

    data: dict[str, list[dict]] = {}
    for method in _STYLE:
        path = args.results / f"{method}_{args.size}" / "sweep.jsonl"
        if path.exists():
            data[method] = _load(path)
        else:
            print(f"missing: {path}")

    fixed: dict[str, dict[int, float]] = {}
    for method, entries in data.items():
        sampler = _FIXED_STEP_SAMPLER[method]
        fixed[method] = {e["nfe"]: e["fid"] for e in entries if e["sampler"] == sampler}

    plt.style.use(["science", "no-latex"])
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    for method, (color, label) in _STYLE.items():
        if method not in data:
            continue
        sampler = _FIXED_STEP_SAMPLER[method]
        xs = sorted(fixed[method])
        ys = [fixed[method][x] for x in xs]
        if xs:
            ax_top.plot(xs, ys, marker="o", color=color, label=f"{label} ({sampler})")
        for e in data[method]:
            if e["sampler"] != sampler:
                ax_top.axhline(e["fid"], color=color, linestyle="--", alpha=0.5,
                               label=f"{label} ({e['sampler']})")

    ax_top.set_xscale("log")
    ax_top.set_ylabel("FID")
    ax_top.set_title(f"CIFAR-10 ({args.size})")
    ax_top.legend()
    ax_top.grid(True, which="both", alpha=0.3)

    common = sorted(set(fixed.get("ddpm", {})) & set(fixed.get("fm", {})))
    if common:
        rel = [100 * (fixed["ddpm"][n] - fixed["fm"][n]) / fixed["ddpm"][n] for n in common]
        ax_bot.plot(common, rel, marker="o", color="black")
        ax_bot.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax_bot.set_ylabel("FM advantage (\\%)")  # (DDPM - FM) / DDPM
    ax_bot.set_xlabel("NFE")
    ax_bot.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
