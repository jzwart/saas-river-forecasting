"""FullGraphTemporalDataset — full-graph temporal windows over precomputed grids.

Unlike the released dataset (which globbed per-node CSVs and filled X_all with
nested Python loops), this reads the dense arrays built by prepare_data.py and
assembles each window's input as:

    X_window = concat([ X_time[window]        (window_len, N, 20),
                        X_static broadcast    (window_len, N, 17) ], axis=-1)

so every node's input vector = [drivers, obs-lags, MaxDepth, month, day, statics].
Windows are assigned to train/val via window_split_map (by window_index).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .windows import WindowSpec, generate_windows


class FullGraphTemporalDataset(Dataset):
    def __init__(self, X_time, X_static, y_all, spec: WindowSpec,
                 window_indices: list[int] | None = None):
        self.X_time = X_time            # (T, N, 20) float32
        self.X_static = X_static        # (N, 17)    float32
        self.y_all = y_all              # (T, N, 2)  float32
        self.spec = spec
        self.T, self.N, _ = X_time.shape

        all_windows = generate_windows(self.T, spec)
        if window_indices is None:
            self.windows = all_windows
            self.window_ids = list(range(len(all_windows)))
        else:
            self.windows = [all_windows[i] for i in window_indices]
            self.window_ids = list(window_indices)

        # Broadcastable static block: (1, N, 17)
        self._static_b = X_static[None, :, :]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start, end = self.windows[idx]
        xt = self.X_time[start:end]                       # (wl, N, 20)
        xs = np.broadcast_to(self._static_b, (xt.shape[0], self.N, self.X_static.shape[1]))
        X = np.concatenate([xt, xs], axis=-1)             # (wl, N, 37)
        y = self.y_all[start:end]                         # (wl, N, 2)
        return {
            "X": torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
            "y": torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)),
            "window_id": self.window_ids[idx],
            "start": start,
            "end": end,
        }


def collate_full_graph(batch):
    return {
        "X": torch.stack([b["X"] for b in batch], dim=0),   # (B, wl, N, 37)
        "y": torch.stack([b["y"] for b in batch], dim=0),   # (B, wl, N, 2)
        "window_id": [b["window_id"] for b in batch],
        "start": [b["start"] for b in batch],
        "end": [b["end"] for b in batch],
    }


def load_split_indices(split_map_path) -> dict[str, list[int]]:
    """Return {'train': [...], 'val': [...]} window indices from the split map."""
    df = pd.read_csv(split_map_path)
    return {
        "train": df.loc[df["split"] == "train", "window_index"].tolist(),
        "val": df.loc[df["split"] == "val", "window_index"].tolist(),
    }
