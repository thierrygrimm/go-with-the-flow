"""Plot FID vs NFE comparing DDPM and FM, plus a delta panel for FM's advantage."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401


# DDIM/Euler are the fixed-step samplers we sweep NFE on. ancestral/RK45 are paper baselines.
_FIXED_STEP_SAMPLER = {"ddpm": "ddim", "fm": "euler"}
_STYLE = {
    "ddpm": ("#0072B2", "DDPM"),   # Okabe-Ito blue
    "fm":   ("#D55E00", "FM"),     # Okabe-Ito vermilion
}
_DELTA_COLOR = "#525252"           # charcoal


def _load(path: Path) -> list[dict]:
    """Read sweep.jsonl, dropping legacy entries that pre-date the `nfe` / `fid` schema."""
    entries: list[dict] = []
    dropped = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if "nfe" not in e or "fid" not in e or "sampler" not in e:
            dropped += 1
            continue
        entries.append(e)
    if dropped:
        print(f"  {path}: dropped {dropped} legacy entries (missing nfe/fid/sampler)")
    return entries


def _plot_fid_vs_nfe(data, fixed, out: Path) -> None:
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
        # Paper sampler: thin dashed FID-level guide + star marker at its actual NFE
        # (ancestral runs at T=1000; RK45 is adaptive and typically lands ~145).
        for e in data[method]:
            if e["sampler"] != sampler:
                ax_top.axhline(e["fid"], color=color, linestyle=":", alpha=0.4,
                               linewidth=1.0)
                ax_top.scatter([e["nfe"]], [e["fid"]], color=color, marker="*",
                               s=110, edgecolors="black", linewidths=0.6, zorder=4,
                               label=f"{label} ({e['sampler']}, NFE={e['nfe']})")
    ax_top.set_xscale("log")
    ax_top.set_ylabel("FID")
    ax_top.set_title("DDPM vs Flow Matching on CIFAR-10")
    ax_top.legend()
    ax_top.grid(True, which="both", alpha=0.3)

    common = sorted(set(fixed.get("ddpm", {})) & set(fixed.get("fm", {})))
    if common:
        rel = [100 * (fixed["ddpm"][n] - fixed["fm"][n]) / fixed["ddpm"][n] for n in common]
        ax_bot.plot(common, rel, marker="o", color=_DELTA_COLOR)
        ax_bot.axhline(0, color="#a0a0a0", linestyle="--", alpha=0.7)
        ax_bot.set_ylabel("FM advantage (\\%)")
    ax_bot.set_xlabel("NFE")
    ax_bot.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"saved: {out}")


def _plot_fid_vs_wallclock(data, out: Path) -> bool:
    """Returns True if any entries had `wall_s` and a plot was produced."""
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for method, (color, label) in _STYLE.items():
        if method not in data:
            continue
        sampler = _FIXED_STEP_SAMPLER[method]
        pts = sorted(
            (e["wall_s"], e["fid"]) for e in data[method]
            if e["sampler"] == sampler and "wall_s" in e
        )
        if pts:
            ws, fs = zip(*pts)
            ax.plot(ws, fs, marker="o", color=color, label=f"{label} ({sampler})")
            plotted = True
        for e in data[method]:
            if e["sampler"] != sampler and "wall_s" in e:
                ax.scatter([e["wall_s"]], [e["fid"]], color=color, marker="*", s=200,
                           edgecolors="black", linewidths=0.8, zorder=4,
                           label=f"{label} ({e['sampler']}, NFE={e['nfe']})")
                plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xscale("log")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("FID")
    ax.set_title("FID vs wall-clock on CIFAR-10")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"saved: {out}")
    return True


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
    _plot_fid_vs_nfe(data, fixed, out)
    _plot_fid_vs_wallclock(data, args.results / f"fid_vs_wallclock_{args.size}.png")


if __name__ == "__main__":
    main()
