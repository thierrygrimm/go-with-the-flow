# go-with-the-flow

Course project, group **A1**: DDPM vs. Flow Matching on CIFAR-10.

## Setup

    make run_bash         # build the image, drop into /workspace
    make help             # list all targets

On driver-535 hosts (no CUDA 12.8 forward-compat) override the base image:

    make _build BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

Local (no Docker, IDE only):

    pip install torch torchvision
    make install

## Smoke test

Train one method for 500 steps, then compute FID:

    make smoke_docker METHOD=ddpm SIZE=small
    make smoke_docker METHOD=fm   SIZE=paper

`METHOD={ddpm,fm}` and `SIZE={small,paper}` (small ~7M params, paper ~35M).
First run downloads CIFAR-10 (~170 MB) and InceptionV3 weights for FID (~95 MB).

## Layout

`src/` is on `PYTHONPATH`.

| Path                  | Role                                              |
|-----------------------|---------------------------------------------------|
| `src/methods/base.py` | `GenerativeMethod` ABC                            |
| `src/methods/unet.py` | Shared U-Net backbone (`small`, `paper` sizes)    |
| `src/methods/ddpm.py` | Noise-prediction DDPM + DDIM sampler              |
| `src/methods/flow.py` | Conditional flow matching + Euler sampler        |
| `src/data/`           | `DatasetBuilder` ABC + CIFAR-10                   |
| `src/metrics/`        | `Metric` ABC + FID                                |
| `src/train.py`        | Training loop                                     |
| `src/evaluate.py`     | Feed real + generated batches to metrics          |
| `src/smoke.py`        | End-to-end smoke entry point                      |
| `src/config.py`       | Run config (dataclasses)                          |

## Open

- Long training to reproduce paper FID (~3.17 DDPM / ~6.35 FM on CIFAR-10).
- Comparison axis: fixed-NFE / wall-clock / seed stability.
