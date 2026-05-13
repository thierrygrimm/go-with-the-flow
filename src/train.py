import torch
from torch.utils.data import DataLoader

from methods.base import GenerativeMethod


def train(
    method: GenerativeMethod,
    train_loader: DataLoader,
    *,
    n_steps: int,
    lr: float = 2e-4,
    log_every: int = 50,
    device: torch.device = torch.device("cpu"),
) -> GenerativeMethod:
    method.to(device).train()
    opt = torch.optim.AdamW(method.parameters(), lr=lr)

    step = 0
    while step < n_steps:
        for batch, _ in train_loader:
            loss = method.training_loss(batch.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % log_every == 0:
                print(f"step {step}/{n_steps}  loss {loss.item():.4f}")
            if step >= n_steps:
                break
    return method
