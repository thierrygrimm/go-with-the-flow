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

## Final paper-FID

Loads `results/<method>_<size>/best.pt`, swaps EMA in, samples with the paper sampler
(ancestral for DDPM, RK45 for FM), and computes FID over 50k samples:

    make eval_server METHOD=ddpm SIZE=paper
    make eval_server METHOD=fm SIZE=paper

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
| `src/eval.py`         | Final paper-FID entry point                       |
| `src/config.py`       | Smoke run config                                  |

## Open

- Long training to reproduce paper FID (~3.17 DDPM / ~6.35 FM on CIFAR-10).
- Comparison axis: fixed-NFE / wall-clock / seed stability.
