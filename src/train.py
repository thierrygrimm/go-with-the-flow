import json
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader

from ema import EMA
from methods.base import GenerativeMethod


def train(
    method: GenerativeMethod,
    train_loader: DataLoader,
    *,
    n_steps: int,
    lr: float = 2e-4,
    log_every: int = 50,
    device: torch.device = torch.device("cpu"),
    ema: EMA | None = None,
    grad_clip: float | None = None,
    warmup_steps: int = 0,
    eval_every: int = 0,
    eval_fn: Callable[[GenerativeMethod], float] | None = None,
    ckpt_dir: Path | None = None,
    resume_from: Path | None = None,
) -> GenerativeMethod:
    method.to(device).train()
    opt = torch.optim.Adam(method.parameters(), lr=lr)
    if ema is not None:
        ema.to(device)

    log_path = ckpt_dir / "log.jsonl" if ckpt_dir else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_fid = float("inf")
    start_step = 0
    if resume_from is not None and resume_from.exists():
        ckpt = torch.load(resume_from, map_location=device)
        method.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
            ema.to(device)
        start_step = ckpt["step"]
        best_fid = ckpt.get("best_fid", float("inf"))
        print(f"resumed from step {start_step}")

    step = start_step
    while step < n_steps:
        for batch, _ in train_loader:
            batch = batch.to(device, non_blocking=True)

            lr_now = lr * min(1.0, (step + 1) / max(warmup_steps, 1))
            for g in opt.param_groups:
                g["lr"] = lr_now

            loss = method.training_loss(batch)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(method.parameters(), grad_clip)
            opt.step()

            if ema is not None:
                ema.update(method, step=step)
            step += 1

            if step % log_every == 0:
                print(f"step {step}/{n_steps}  loss {loss.item():.4f}  lr {lr_now:.2e}")
                _log(log_path, step=step, loss=loss.item(), lr=lr_now)

            if eval_every and eval_fn and step % eval_every == 0:
                ctx = ema.swap_in(method) if ema is not None else nullcontext()
                with ctx:
                    fid = eval_fn(method)
                print(f"eval @ {step}: fid={fid:.2f}")
                _log(log_path, step=step, eval_fid=fid)
                if ckpt_dir is not None:
                    _save(ckpt_dir / "latest.pt", method, ema, opt, step, best_fid)
                    if fid < best_fid:
                        best_fid = fid
                        _save(ckpt_dir / "best.pt", method, ema, opt, step, best_fid)

            if step >= n_steps:
                break

    if ckpt_dir is not None:
        _save(ckpt_dir / "latest.pt", method, ema, opt, step, best_fid)

    return method


def _save(path: Path, method, ema, opt, step: int, best_fid: float) -> None:
    state = {"model": method.state_dict(), "opt": opt.state_dict(), "step": step, "best_fid": best_fid}
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, path)


def _log(path: Path | None, **kw) -> None:
    if path is not None:
        with path.open("a") as f:
            f.write(json.dumps(kw) + "\n")
    if wandb.run is not None:
        payload = {k: v for k, v in kw.items() if k != "step"}
        wandb.log(payload, step=kw.get("step"))
