"""Evaluation: NLL, sample grid, and FID-vs-NFE sweep with wall-clock time.

Loads <method>_<size>/best.pt, swaps EMA in, then writes to that same directory:
  1. NLL on the test set (bits/dim)        -> nll.json
  2. Sample grid from the paper sampler    -> samples.png
  3. FID + wall-clock per sampler config   -> sweep.jsonl  (resume-safe)

DDPM NLL: stochastic single-timestep L_VLB (Ho et al. eq 5, fixedlarge variance);
ignores the L_T prior-KL and L_0 discrete-decoder terms, adds +log2(128) to convert
continuous bpd to discrete bits/dim.

FM NLL: Hutchinson trace of the divergence along a reverse-time ODE (dopri5,
rtol=atol=1e-5), same +log2(128) convention.
"""
import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torchdiffeq
import torchvision.utils as vutils

from data.cifar10 import CIFAR10
from ema import EMA
from evaluate import evaluate
from methods import METHODS, SIZES, build
from methods.ddpm import DDPM
from methods.flow import FlowMatching
from metrics.fid import FIDMetric


_DEQUANT_BITS = math.log2(128)  # +7 bpd: [-1, 1] from uint8 images

# (sampler, steps) per method. steps=None for adaptive samplers.
SWEEPS: dict[str, list[tuple[str, int | None]]] = {
    "ddpm": [
        ("ddim", 5), ("ddim", 10), ("ddim", 20), ("ddim", 50),
        ("ddim", 100), ("ddim", 250), ("ddim", 1000),
        ("ancestral", None),
    ],
    "fm": [
        ("euler", 5), ("euler", 10), ("euler", 20), ("euler", 50),
        ("euler", 100), ("euler", 250), ("euler", 1000),
        ("rk45", None),
    ],
}


@torch.no_grad()
def _ddpm_bpd(method: DDPM, x: torch.Tensor) -> torch.Tensor:
    B = x.shape[0]
    T = method.n_timesteps
    dim = x[0].numel()
    t = torch.randint(0, T, (B,), device=x.device)
    noise = torch.randn_like(x)
    ab = method.alpha_bars[t][:, None, None, None]
    beta = method.betas[t][:, None, None, None]
    alpha = 1.0 - beta
    x_t = ab.sqrt() * x + (1 - ab).sqrt() * noise
    eps_pred = method.model(x_t, t)
    weight = beta / (2 * alpha * (1 - ab))
    nll_nats = T * (weight * (eps_pred - noise) ** 2).sum(dim=(1, 2, 3))
    return nll_nats / (dim * math.log(2)) + _DEQUANT_BITS


def _fm_bpd(method: FlowMatching, x: torch.Tensor, *,
            rtol: float = 1e-3, atol: float = 1e-3) -> tuple[torch.Tensor, int]:
    """Returns (bpd_per_sample, nfe_used). dopri5 at tight rtol/atol on a 3072-dim image
    ODE explodes in NFE; defaults loosened from 1e-5 to 1e-3 (bpd changes in the 3rd
    decimal but compute drops ~10-100x)."""
    B = x.shape[0]
    dim = x[0].numel()
    device = x.device
    nfe = [0]

    def f_aug(t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        nfe[0] += 1
        x_t, _ = state
        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_(True)
            v = method.model(x_t, (t * 1000.0).expand(B))
            eps = torch.randn_like(x_t)
            v_eps = (v * eps).sum()
            grad = torch.autograd.grad(v_eps, x_t)[0]
            div = (grad * eps).flatten(1).sum(1)
        return v.detach(), div.detach()

    t_span = torch.tensor([1.0, 0.0], device=device)
    state0 = (x, torch.zeros(B, device=device))
    with torch.no_grad():  # don't build the outer autograd graph through every odeint step
        sol = torchdiffeq.odeint(f_aug, state0, t_span, method="dopri5", rtol=rtol, atol=atol)
    z = sol[0][-1]
    lndet = sol[1][-1]
    log_p_z = -0.5 * (z ** 2).flatten(1).sum(1) - 0.5 * dim * math.log(2 * math.pi)
    bpd = -(log_p_z + lndet) / (dim * math.log(2)) + _DEQUANT_BITS
    return bpd, nfe[0]


def compute_nll(method, data, *, n_samples: int, batch_size: int, device,
                out_path: Path, fm_rtol: float = 1e-3, fm_atol: float = 1e-3) -> None:
    is_fm = isinstance(method, FlowMatching)
    loader = data.eval_loader(batch_size)
    print(f"[nll] n={n_samples}"
          + (f"  (FM: dopri5 rtol={fm_rtol} atol={fm_atol})" if is_fm else ""))
    bpds: list[torch.Tensor] = []
    seen = 0
    for batch, _ in loader:
        n = min(batch.shape[0], n_samples - seen)
        if is_fm:
            bpd, nfe = _fm_bpd(method, batch[:n].to(device), rtol=fm_rtol, atol=fm_atol)
            tag = f" nfe={nfe}"
        else:
            bpd = _ddpm_bpd(method, batch[:n].to(device))
            tag = ""
        bpds.append(bpd.detach().cpu())
        seen += n
        print(f"  {seen}/{n_samples}  bpd={bpd.mean().item():.4f}{tag}", flush=True)
        if seen >= n_samples:
            break
    all_bpd = torch.cat(bpds)
    summary = {
        "n": seen,
        "bpd_mean": all_bpd.mean().item(),
        "bpd_std": all_bpd.std().item(),
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"  -> {out_path}: {summary['bpd_mean']:.4f} +/- {summary['bpd_std']:.4f} bits/dim")


def save_grid(method, *, n: int, nrow: int, device, out_path: Path, seed: int) -> None:
    sampler = type(method).paper_sampler
    torch.manual_seed(seed)
    with torch.no_grad():
        samples = method.sample(n, device=device, sampler=sampler)
    samples = (samples.clamp(-1, 1) + 1) / 2
    vutils.save_image(samples, out_path, nrow=nrow)
    print(f"[grid] -> {out_path}  (sampler={sampler}, n={n})")


def run_sweep(method, data, *, n_samples: int, sample_batch: int, device,
              out_path: Path, method_name: str) -> None:
    # Read existing entries; legacy ones missing `wall_s` (from before wall-clock
    # tracking) are dropped from the file and re-run.
    done: set[tuple[str, int | None]] = set()
    kept: list[dict] = []
    n_legacy = 0
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if "wall_s" not in e:
                n_legacy += 1
                continue
            kept.append(e)
            s = e["sampler"]
            done.add((s, None) if s in ("ancestral", "rk45") else (s, e["nfe"]))
        if n_legacy:
            out_path.write_text("".join(json.dumps(e) + "\n" for e in kept))
            print(f"[sweep] dropped {n_legacy} legacy entries (no wall_s) for re-run")
    remaining = [(s, st) for (s, st) in SWEEPS[method_name] if (s, st) not in done]
    print(f"[sweep] {len(done)} done, {len(remaining)} remaining -> {out_path}")
    if not remaining:
        return

    fid_metric = FIDMetric(device=device)
    eval_loader = data.eval_loader(sample_batch)
    with out_path.open("a") as f:
        for sampler, steps in remaining:
            kwargs: dict = {"sampler": sampler}
            if steps is not None:
                kwargs["steps"] = steps
            print(f"  {sampler} steps={steps} ...", end=" ", flush=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            results = evaluate(
                method, eval_loader, [fid_metric],
                n_samples=n_samples,
                sample_batch=sample_batch,
                device=device,
                sample_kwargs=kwargs,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall_s = time.perf_counter() - t0
            if sampler == "ancestral":
                nfe = method.n_timesteps
            elif sampler == "rk45":
                nfe = method.last_nfe  # counted inside FlowMatching.sample
            else:
                nfe = steps
            entry = {
                "method": method_name,
                "sampler": sampler,
                "nfe": nfe,
                "n_samples": n_samples,
                "fid": results["FIDMetric"],
                "wall_s": wall_s,
            }
            print(f"nfe={nfe} FID={entry['fid']:.4f} wall={wall_s:.1f}s")
            f.write(json.dumps(entry) + "\n")
            f.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(METHODS), default="ddpm")
    parser.add_argument("--size", choices=list(SIZES), default="paper")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--t-embed", default="sinusoidal", choices=["sinusoidal", "fourier"])
    parser.add_argument("--n-samples-fid", type=int, default=10_000)
    parser.add_argument("--n-samples-nll", type=int, default=1024)
    parser.add_argument("--n-samples-grid", type=int, default=64)
    parser.add_argument("--sample-batch", type=int, default=128)
    parser.add_argument("--nll-batch", type=int, default=64)
    parser.add_argument("--grid-nrow", type=int, default=8)
    parser.add_argument("--grid-seed", type=int, default=0)
    parser.add_argument("--skip-nll", action="store_true")
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--nll-rtol", type=float, default=1e-3,
                        help="FM-only: dopri5 relative tolerance for NLL ODE")
    parser.add_argument("--nll-atol", type=float, default=1e-3,
                        help="FM-only: dopri5 absolute tolerance for NLL ODE")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt or Path(f"./results/{args.method}_{args.size}/best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    out_dir = ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = CIFAR10()
    method = build(args.method, args.size, data.shape, t_embed=args.t_embed).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    method.load_state_dict(ckpt["model"])

    ema: EMA | None = None
    if "ema" in ckpt:
        ema = EMA(method).to(device)
        ema.load_state_dict(ckpt["ema"])

    method.eval()
    print(f"eval {args.method} ({args.size}) from {ckpt_path}")

    ctx = ema.swap_in(method) if ema is not None else nullcontext()
    with ctx:
        if not args.skip_nll:
            compute_nll(method, data, n_samples=args.n_samples_nll,
                        batch_size=args.nll_batch, device=device,
                        out_path=out_dir / "nll.json",
                        fm_rtol=args.nll_rtol, fm_atol=args.nll_atol)
        if not args.skip_grid:
            save_grid(method, n=args.n_samples_grid, nrow=args.grid_nrow,
                      device=device, out_path=out_dir / "samples.png",
                      seed=args.grid_seed)
        if not args.skip_sweep:
            run_sweep(method, data, n_samples=args.n_samples_fid,
                      sample_batch=args.sample_batch, device=device,
                      out_path=out_dir / "sweep.jsonl",
                      method_name=args.method)


if __name__ == "__main__":
    main()
