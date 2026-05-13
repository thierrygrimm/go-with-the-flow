# go-with-the-flow

Course project, group **A1**: DDPM vs. Flow Matching on CIFAR-10.

## Setup

Docker (preferred):

    make run_bash         # builds the image and drops you in /workspace

Or local:

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    make install

`make help` lists every target.

## Smoke test

End-to-end pipeline check using a random-noise placeholder method:

    make smoke_docker     # inside docker
    make smoke            # local

First run downloads CIFAR-10 (~170 MB) and InceptionV3 weights for FID (~95 MB).

## Layout

`src/` is on `PYTHONPATH`.

| Path                | Role                                                       |
|---------------------|------------------------------------------------------------|
| `src/methods/`      | `GenerativeMethod` base + subclasses                       |
| `src/data/`         | `DatasetBuilder` base + CIFAR-10                           |
| `src/metrics/`      | `Metric` base + FID                                        |
| `src/train.py`      | Training loop                                              |
| `src/evaluate.py`   | Feed real + generated batches to metrics                   |
| `src/smoke.py`      | End-to-end smoke entry point                               |
| `src/config.py`     | Run config (dataclasses)                                   |

## Open

- DDPM and Flow Matching: add as `GenerativeMethod` subclasses in `src/methods/`.
- Extension axis: fixed-NFE / wall-clock / seed stability.
