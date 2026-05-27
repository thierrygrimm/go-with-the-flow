"""FID vs NFE with every sampler overlaid -- isolates the sampler from the model.

Reads results/<method>_<size>/sweep.jsonl (after `make eval` this holds the
1st-order samplers, Heun, and the adaptive/ancestral baselines) and draws one
line per fixed-step sampler plus a star per adaptive/ancestral point. If the
2nd-order Heun integrator closes the DDPM<->FM gap at matched NFE, the advantage
was path curvature that Euler handled badly; if it persists, it is the learned
vector field itself.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

_COLOR = {"ddpm": "#0072B2", "fm": "#D55E00"}  # Okabe-Ito, same as plot.py
_STAR = {"ancestral", "rk45"}                  # adaptive / paper baselines


def _load(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            if {"nfe", "fid", "sampler"} <= e.keys():
                out.append(e)
    return out


def _plot_fid_vs_nfe(data: dict[str, list[dict]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method, color in _COLOR.items():
        by_sampler: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for e in data.get(method, []):
            by_sampler[e["sampler"]].append((e["nfe"], e["fid"]))
        for sampler, pts in by_sampler.items():
            xs, ys = zip(*sorted(pts))
            label = f"{method.upper()} ({sampler})"
            if sampler in _STAR:
                ax.scatter(xs, ys, color=color, marker="*", s=160, zorder=4,
                           edgecolors="black", linewidths=0.6, label=label)
            else:  # 1st-order solid, Heun dashed+square so the pair is easy to compare
                ax.plot(xs, ys, color=color, marker="s" if sampler == "heun" else "o",
                        linestyle="--" if sampler == "heun" else "-", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("NFE")
    ax.set_ylabel("FID")
    ax.set_title("Sampler vs model: FID vs NFE on CIFAR-10")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"saved: {out}")


def _plot_decomp(data: dict[str, list[dict]], out: Path) -> None:
    """Stacked FID = mean term (fidelity) + cov term (diversity), best config per sampler."""
    rows = []  # (method, sampler, mean_term, cov_term)
    for method in _COLOR:
        best: dict[str, dict] = {}
        for e in data.get(method, []):
            if "fid_mean_term" not in e:
                continue
            if e["sampler"] not in best or e["fid"] < best[e["sampler"]]["fid"]:
                best[e["sampler"]] = e
        for sampler, e in best.items():
            rows.append((method, sampler, e["fid_mean_term"], e["fid_cov_term"]))
    if not rows:
        print("no fid_mean_term/fid_cov_term in sweeps yet (re-run the sweep to populate)")
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(rows)), 4.5))
    x = range(len(rows))
    means = [r[2] for r in rows]
    covs = [r[3] for r in rows]
    ax.bar(x, means, color="#4C72B0", label="mean term (centroid offset ~ fidelity)")
    ax.bar(x, covs, bottom=means, color="#DD8452", label="cov term (spread mismatch ~ diversity)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{m.upper()}\n({s})" for (m, s, _, _) in rows])
    ax.set_ylabel("FID contribution")
    ax.set_title("Where each method's FID comes from")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("./results"))
    parser.add_argument("--size", default="paper")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.results / f"fid_vs_nfe_samplers_{args.size}.png"

    data: dict[str, list[dict]] = {}
    for method in _COLOR:
        path = args.results / f"{method}_{args.size}" / "sweep.jsonl"
        if path.exists():
            data[method] = _load(path)
        else:
            print(f"missing: {path}")

    plt.style.use(["science", "no-latex"])
    out.parent.mkdir(parents=True, exist_ok=True)
    _plot_fid_vs_nfe(data, out)
    _plot_decomp(data, args.results / f"fid_decomp_{args.size}.png")


if __name__ == "__main__":
    main()
