"""
cmems_dataset_ar.py

A sliding-window dataset that returns *multi-step* targets for
autoregressive curriculum training.

Place alongside your existing cmems_dataset.py:
    src/data/cmems_dataset_ar.py

Usage:
    from src.data.cmems_dataset_ar import SlidingWindowMultiStep

    ds = SlidingWindowMultiStep(uv_norm, seq_len=12, max_horizon=12)
    X, Y_seq = ds[i]
    # X:     (seq_len, 2, H, W)   — input window
    # Y_seq: (max_horizon, 2, H, W) — next 1..max_horizon ground-truth frames
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


class SlidingWindowMultiStep(Dataset):
    """
    Given a normalised array  uv_norm  of shape (T, 2, H, W),
    yields (X, Y_seq) pairs where:
        X      = uv_norm[i : i+seq_len]            -> (seq_len, 2, H, W)
        Y_seq  = uv_norm[i+seq_len : i+seq_len+H]  -> (max_horizon, 2, H, W)

    The trainer can slice Y_seq[:ar_steps] to use only the
    steps required by the current curriculum stage.
    """

    def __init__(self, uv: np.ndarray, seq_len: int = 12, max_horizon: int = 12):
        super().__init__()
        self.uv = uv                     # (T, 2, H, W)
        self.seq_len = seq_len
        self.max_horizon = max_horizon
        # Number of valid starting indices
        self.n_samples = len(uv) - seq_len - max_horizon

        if self.n_samples <= 0:
            raise ValueError(
                f"Dataset too short: T={len(uv)}, need at least "
                f"seq_len({seq_len}) + max_horizon({max_horizon}) + 1 = "
                f"{seq_len + max_horizon + 1} frames."
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        s = idx
        e = s + self.seq_len

        X = self.uv[s:e]                          # (seq_len, 2, H, W)
        Y = self.uv[e : e + self.max_horizon]     # (max_horizon, 2, H, W)

        return (
            torch.from_numpy(X).float(),
            torch.from_numpy(Y).float(),
        )
