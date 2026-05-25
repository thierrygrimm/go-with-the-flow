# go-with-the-flow

Course project, group **A1**: DDPM vs. Flow Matching on CIFAR-10.

## Setup

Pick a docker target by host CUDA driver:

    make run_bash_local      # CUDA 12.1 (driver <= 535)
    make run_bash_server     # CUDA 12.8 (Blackwell-capable)
    make help                # list all targets

Local (no Docker, IDE only):

    pip install torch torchvision
    make install

## Smoke test

Train one method for 500 steps, then compute FID:

    make smoke_local METHOD=ddpm SIZE=small
    make smoke_server METHOD=fm SIZE=paper

`METHOD={ddpm,fm}` and `SIZE={small,paper}` (small ~9M params, paper ~35.7M, matching Ho et al.).
First run downloads CIFAR-10 (~170 MB) and InceptionV3 weights for FID (~95 MB).

## Training

Paper-exact training: fp32, Adam(lr=2e-4, wd=0), EMA 0.9999, grad clip 1.0, LR warmup
5000, hflip aug, batch 128, dropout 0.1, T=1000, linear beta schedule. Periodic FID +
latest/best checkpoint by held-out FID. Results land in `results/<method>_<size>/`.

    make train_local METHOD=ddpm SIZE=paper N_STEPS=100000
    make train_server METHOD=fm SIZE=paper N_STEPS=100000

Resume an interrupted run from `results/<method>_<size>/latest.pt`:

    make train_local METHOD=ddpm SIZE=paper RESUME=--resume

W&B is on by default (project `go-with-the-flow`); set `WANDB_API_KEY` in your shell
or set `WANDB_MODE=disabled` to skip.

Periodic FID during training uses fast samplers (DDIM for DDPM, Euler for FM); the
final paper FID is computed separately via `make eval_*` (see below).

## Evaluation

One command runs all metrics against `results/<method>_<size>/best.pt`:

    make eval_server METHOD=ddpm SIZE=paper
    make eval_server METHOD=fm   SIZE=paper

This loads the checkpoint, swaps EMA in, then writes (in order):

- `nll.json` - bits/dim on the test set (`N_SAMPLES_NLL=1024` by default)
- `samples.png` - 64-image grid from the paper sampler (ancestral / RK45)
- `sweep.jsonl` - FID + wall-clock per `(sampler, steps)` config, `N_SAMPLES_FID=10000`
  per config, 8 configs per method (DDIM/Euler at 5/10/20/50/100/250/1000 NFE plus
  the adaptive paper sampler). Resume-safe: kill any time, re-run to continue from
  the last completed config.

Skip individual stages with `--skip-nll`, `--skip-grid`, `--skip-sweep`.

After both methods finish, build the comparison plots:

    make plot_server SIZE=paper
    # -> results/fid_vs_nfe_paper.png       (top: FID vs NFE; bottom: FM advantage %)
    # -> results/fid_vs_wallclock_paper.png (FID vs wall-clock; shows RK45 cost)

Trajectory visualization (noise -> image, low/mid/high NFE side by side):

    make trajectory_server METHOD=ddpm SIZE=paper
    make trajectory_server METHOD=fm   SIZE=paper
    # -> results/<method>_<size>/trajectory.png

## Layout

`src/` is on `PYTHONPATH`.

| Path                  | Role                                              |
|-----------------------|---------------------------------------------------|
| `src/methods/base.py` | `GenerativeMethod` ABC                            |
| `src/methods/unet.py` | Shared U-Net backbone (`small`, `paper` sizes)    |
| `src/methods/ddpm.py` | DDPM (DDIM + ancestral samplers)                  |
| `src/methods/flow.py` | Conditional flow matching (Euler + RK45 samplers) |
| `src/data/`           | `DatasetBuilder` ABC + CIFAR-10                   |
| `src/metrics/`        | `Metric` ABC + FID                                |
| `src/train.py`        | Training loop (EMA, grad clip, warmup, checkpoints) |
| `src/evaluate.py`     | Feed real + generated batches to metrics          |
| `src/ema.py`          | EMA helper                                        |
| `src/smoke.py`        | Smoke entry point (short training run)            |
| `src/run.py`          | Real training entry point                         |
| `src/eval.py`         | Eval entry: NLL + sample grid + FID-vs-NFE sweep  |
| `src/plot.py`         | FID-vs-NFE and FID-vs-wall-clock plots            |
| `src/config.py`       | Smoke run config                                  |

## Experimental setup

**Data**. CIFAR-10 (50k train, 10k test), 32×32 RGB, normalized to [-1, 1].
Random horizontal flip on train; no flip on eval. PyTorch DataLoader with
`pin_memory` + `persistent_workers`.

**Shared backbone**. Ho et al. 2020 DDPM++ U-Net used by both methods for a fair
A/B on the loss. `SIZE=paper`: `base=128`, `ch_mults=(1,2,2,2)`, **35.7M params**
(matches Ho); `SIZE=small`: `base=64`, ~9M params. 2 ResBlocks per down level + 3
per up level (`num_res_blocks+1`); self-attention after every ResBlock at 16×16
only; GroupNorm 32 groups; SiLU; dropout 0.1 (paper preset); 3×3 stride-2
downsample, nearest-exact + 3×3 conv upsample; sinusoidal time embedding
(`(half-1)` denominator, fp32 freqs) → 2-layer MLP (4·base hidden). Init:
Xavier-uniform on every Conv2d/Linear (Ho's `default_init`); zero-init on
ResBlock `conv2`, Attention `proj`, and the final `conv_out` (Ho's
`init_scale=0`).

**DDPM**. T=1000, linear β schedule 1e-4 → 0.02. Training loss is **L_simple**
(`((eps - noise)**2).mean()` — uniform weight across t, no `1/(2σ²)`
reweighting). Samplers: **DDIM (η=0)** for periodic eval and the NFE sweep;
**ancestral** (σ_t=√β_t, "fixedlarge") for the paper-FID reproduction.

**FM**. **ICFM** with the **linear/OT interpolant**, σ_min=1e-4 (Lipman et al.
Table 5); we do *not* use OT batch coupling (Lipman shows same FID 6.35 for
ICFM and OT-CFM on CIFAR). Loss `||v_θ - (x_1 - (1-σ_min)·x_0)||²`, t ∼
Uniform[0,1]. Samplers: **Euler** (fixed-step) for the NFE sweep; adaptive
**RK45** (`dopri5`, rtol=atol=1e-5, ~145 NFE) for the paper-FID reproduction.

**Training (both methods)**. Adam(lr=2e-4, wd=0, default β), batch 128, fp32
(no AMP), gradient clip 1.0, LR linear warmup over 5000 steps. EMA decay 0.9999
with paper warmup ramp `min(decay, (1+step)/(10+step))`. Trained N_STEPS=200K
each (paper: 800K) on a single A6000.

**Evaluation**. FID via torchmetrics (InceptionV3 features, uint8 inputs).
Periodic during training: 1k samples every 5k steps with the fast sampler
(DDIM-100 / Euler-50), used for the `best.pt` checkpoint. NFE-sweep deliverable
(`eval.py`): 10k samples per config, 8 configs per method, with wall-clock time
recorded alongside FID so we can plot FID-vs-wall-clock (RK45's adaptive
integration is far more expensive per call than Euler at matched NFE).

**NLL**. Bits/dim on the CIFAR-10 test set (computed in `eval.py`). DDPM uses
Ho et al.'s stochastic L_VLB estimator (single-timestep MSE weighted by
`β / (2α(1-ᾱ))`, scaled by T); FM uses Hutchinson trace estimation of the
divergence along a reverse-time ODE integration (dopri5, rtol=atol=1e-5). Both
add a `+log₂(128)≈7` dequantization offset to convert continuous bpd to discrete
bits/dim. We omit the small L_T prior-KL and L_0 discrete-decoder terms in
DDPM's estimator.

**Hardware**. Single NVIDIA RTX A6000 (48GB VRAM). Docker on CUDA 12.8.

## Ablations

- **DDPM noise schedule** (Improved DDPM cosine vs paper linear): train with
  `python -m run --method ddpm --size paper --n-steps N --schedule cosine`.
- **FM `sigma_min`**: `... --method fm --sigma-min 0.0` to disable, or any other value.
- **Model size**: `SIZE=small` (~9M params) trains 4x faster than `SIZE=paper`.
- **Time embedding** (Tancik et al. random Fourier features vs Ho et al. sinusoidal):
  `... --t-embed fourier`. Pass the same flag to `eval.py` when evaluating a
  checkpoint that was trained with `fourier`.
