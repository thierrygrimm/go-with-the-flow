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
        ("ddim", 100), ("ddim", 250),
        ("ancestral", None),
        # Heun (2nd-order): steps chosen so NFE=2*steps-1 lands near the DDIM grid.
        ("heun", 3), ("heun", 5), ("heun", 10), ("heun", 25), ("heun", 50), ("heun", 125),
    ],
    "fm": [
        ("euler", 5), ("euler", 10), ("euler", 20), ("euler", 50),
        ("euler", 100), ("euler", 250),
        ("rk45", None),
        # Heun (2nd-order): steps chosen so NFE=2*steps lands on the Euler grid.
        ("heun", 3), ("heun", 5), ("heun", 10), ("heun", 25), ("heun", 50), ("heun", 125),
    ],
}


@torch.no_grad()
def _ddpm_bpd(method: DDPM, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Full DDPM L_VLB: iterates over ALL T timesteps per image + closed-form L_T, L_0.

    L_VLB = L_T  +  sum_{t=2..T} L_{t-1}  +  L_0
      L_T : prior KL ( q(x_T|x_0) || N(0, I) ) -- closed form, tiny
      L_{t-1}: per-step KL for t=2..T, fixedsmall sigma^2 = beta-tilde_t (tightest bound)
               -> weight = beta_t / (2 alpha_t (1 - alphabar_{t-1})), times ||eps - eps_theta||^2
               one noise sample per (image, t); summed across all T-1 t's (no Monte Carlo
               magnification factor, so variance is ~T^2 lower than single-t MC).
      L_0 : discrete decoder. Approximated by continuous Gaussian density at x_0 under
               N(mu_theta(x_1, 0), beta_1 I) + log(2/255) per dim for the dequant bin width.

    Cost: ~T = 1000 forward passes per batch, ~20 min for 1024 samples on an A6000.

    Returns (bpd_per_image, components) where components are mean per-image bpd
    contributions of L_T / L_kl / L_0 for diagnostics.
    """
    B = x.shape[0]
    T = method.n_timesteps
    dim = x[0].numel()
    device = x.device
    log2_dim = dim * math.log(2)

    betas = method.betas              # [T]; betas[i] is beta_{i+1} in 1-indexed Ho notation
    alpha_bars = method.alpha_bars    # alpha_bars[i] = alphabar_{i+1}

    # L_T -- prior KL N(sqrt(abar_T) x_0, (1-abar_T) I) || N(0, I), summed over pixels
    ab_T = alpha_bars[-1]
    L_T = 0.5 * ab_T * x.pow(2).flatten(1).sum(1) \
        + 0.5 * dim * (-ab_T - (1 - ab_T).log())

    # sum of L_{t-1} for t = 2..T  (i.e., array indices 1..T-1)
    L_kl = torch.zeros(B, device=device)
    for arr_t in range(1, T):
        noise = torch.randn_like(x)
        ab_t = alpha_bars[arr_t]
        ab_tm1 = alpha_bars[arr_t - 1]
        beta_t = betas[arr_t]
        alpha_t = 1.0 - beta_t
        x_t = ab_t.sqrt() * x + (1 - ab_t).sqrt() * noise
        eps_pred = method.model(x_t, torch.full((B,), arr_t, device=device, dtype=torch.long))
        weight = beta_t / (2 * alpha_t * (1 - ab_tm1))
        L_kl = L_kl + weight * (eps_pred - noise).pow(2).flatten(1).sum(1)

    # L_0 -- decoder. x_1 = sqrt(abar_1) x + sqrt(1-abar_1) noise; mu_theta via Tweedie.
    noise0 = torch.randn_like(x)
    ab_1 = alpha_bars[0]
    beta_1 = betas[0]
    alpha_1 = 1.0 - beta_1
    x_1 = ab_1.sqrt() * x + (1 - ab_1).sqrt() * noise0
    eps_pred_1 = method.model(x_1, torch.zeros(B, device=device, dtype=torch.long))
    mu_theta = (x_1 - beta_1 / (1 - ab_1).sqrt() * eps_pred_1) / alpha_1.sqrt()
    # -log p_continuous(x_0; mu_theta, beta_1) + log(255/2) per dim (dequant)
    L_0 = 0.5 * dim * math.log(2 * math.pi * beta_1.item()) \
        + (x - mu_theta).pow(2).flatten(1).sum(1) / (2 * beta_1) \
        + dim * math.log(255 / 2)

    total_nats = L_T + L_kl + L_0
    bpd = total_nats / log2_dim
    components = {
        "L_T_bpd":  (L_T.mean() / log2_dim).item(),
        "L_kl_bpd": (L_kl.mean() / log2_dim).item(),
        "L_0_bpd":  (L_0.mean() / log2_dim).item(),
    }
    return bpd, components


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
          + (f"  (FM: dopri5 rtol={fm_rtol} atol={fm_atol})"
             if is_fm else f"  (DDPM: full L_VLB, T={method.n_timesteps} per image)"))
    bpds: list[torch.Tensor] = []
    components_acc: list[dict[str, float]] = []
    seen = 0
    t_start = time.perf_counter()
    for batch, _ in loader:
        n = min(batch.shape[0], n_samples - seen)
        if is_fm:
            bpd, nfe = _fm_bpd(method, batch[:n].to(device), rtol=fm_rtol, atol=fm_atol)
            tag = f" nfe={nfe}"
        else:
            bpd, comp = _ddpm_bpd(method, batch[:n].to(device))
            components_acc.append(comp)
            tag = f"  L_T={comp['L_T_bpd']:.3f} L_kl={comp['L_kl_bpd']:.3f} L_0={comp['L_0_bpd']:.3f}"
        bpds.append(bpd.detach().cpu())
        seen += n
        elapsed = time.perf_counter() - t_start
        print(f"  {seen}/{n_samples}  bpd={bpd.mean().item():.4f}  ({elapsed:.0f}s){tag}", flush=True)
        if seen >= n_samples:
            break
    all_bpd = torch.cat(bpds)
    summary: dict = {
        "n": seen,
        "bpd_mean": all_bpd.mean().item(),
        "bpd_std": all_bpd.std().item(),
    }
    if components_acc:
        summary["components_bpd"] = {
            k: sum(c[k] for c in components_acc) / len(components_acc) for k in components_acc[0]
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
            # evaluate() resets metrics only at the start of a call, so the Inception
            # stats are still live here: the FID mean/cov split costs no extra samples.
            comp = fid_metric.compute_components()
            if sampler == "ancestral":
                nfe = method.n_timesteps
            elif sampler in ("rk45", "heun"):
                nfe = method.last_nfe  # counted inside the sampler (2 evals/step for heun)
            else:
                nfe = steps
            entry = {
                "method": method_name,
                "sampler": sampler,
                "nfe": nfe,
                "n_samples": n_samples,
                "fid": results["FIDMetric"],
                "fid_mean_term": comp["fid_mean_term"],
                "fid_cov_term": comp["fid_cov_term"],
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
