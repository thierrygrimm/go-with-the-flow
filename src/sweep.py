"""NFE sweep: evaluate a checkpoint at multiple sampler/steps configurations."""
import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from data.cifar10 import CIFAR10
from ema import EMA
from evaluate import evaluate
from methods import METHODS, SIZES, build
from metrics.fid import FIDMetric


# (sampler, steps) per method. steps=None for samplers that don't take a step count.
SWEEPS: dict[str, list[tuple[str, int | None]]] = {
    "ddpm": [
        ("ddim", 5),
        ("ddim", 10),
        ("ddim", 20),
        ("ddim", 50),
        ("ddim", 100),
        ("ddim", 250),
        ("ddim", 1000),
        ("ancestral", None),
    ],
    "fm": [
        ("euler", 5),
        ("euler", 10),
        ("euler", 20),
        ("euler", 50),
        ("euler", 100),
        ("euler", 250),
        ("euler", 1000),
        ("rk45", None),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(METHODS), default="ddpm")
    parser.add_argument("--size", choices=list(SIZES), default="paper")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--sample-batch", type=int, default=128)
    parser.add_argument("--t-embed", default="sinusoidal", choices=["sinusoidal", "fourier"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt or Path(f"./results/{args.method}_{args.size}/best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    out_path = ckpt_path.parent / "sweep.jsonl"

    data = CIFAR10()
    method = build(args.method, args.size, data.shape, t_embed=args.t_embed).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    method.load_state_dict(ckpt["model"])

    ema: EMA | None = None
    if "ema" in ckpt:
        ema = EMA(method).to(device)
        ema.load_state_dict(ckpt["ema"])

    fid_metric = FIDMetric(device=device)
    eval_loader = data.eval_loader(args.sample_batch)

    # Resume: skip configs already in sweep.jsonl. Dedup by (sampler, steps); for adaptive
    # samplers (steps=None) we treat any existing entry as "done".
    done: set[tuple[str, int | None]] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            s = e["sampler"]
            done.add((s, None) if s in ("ancestral", "rk45") else (s, e["nfe"]))
    remaining = [(s, st) for (s, st) in SWEEPS[args.method] if (s, st) not in done]
    if not remaining:
        print(f"all configs already in {out_path}; nothing to do")
        return

    print(f"sweep {args.method} ({args.size}, n={args.n_samples}) -> {out_path}")
    print(f"  {len(done)} done, {len(remaining)} remaining")
    ctx = ema.swap_in(method) if ema is not None else nullcontext()
    with ctx, out_path.open("a") as f:
        for sampler, steps in remaining:
            kwargs: dict = {"sampler": sampler}
            if steps is not None:
                kwargs["steps"] = steps
            print(f"  {sampler} steps={steps} ...", end=" ", flush=True)
            results = evaluate(
                method, eval_loader, [fid_metric],
                n_samples=args.n_samples,
                sample_batch=args.sample_batch,
                device=device,
                sample_kwargs=kwargs,
            )
            if sampler == "ancestral":
                nfe = method.n_timesteps
            elif sampler == "rk45":
                nfe = method.last_nfe  # counted inside FlowMatching.sample
            else:
                nfe = steps
            entry = {
                "method": args.method,
                "sampler": sampler,
                "nfe": nfe,
                "n_samples": args.n_samples,
                "fid": results["FIDMetric"],
            }
            print(f"nfe={nfe} FID={entry['fid']:.4f}")
            f.write(json.dumps(entry) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
