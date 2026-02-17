from __future__ import annotations
import os, json, random
import numpy as np
import torch

def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(batch, device: torch.device):
    if isinstance(batch, (list, tuple)):
        return [to_device(x, device) for x in batch]
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch

def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def rmse(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean((a - b) ** 2) + eps)
