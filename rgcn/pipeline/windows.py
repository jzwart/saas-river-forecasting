"""Canonical temporal-window generation, shared by make_splits / train / export.

A window spans ``window_len = seq_length + forecast_horizon`` consecutive days.
The model sees the first ``seq_length`` days as history and is scored on the last
``forecast_horizon`` days; the "day-3" forecast is the final day of the window.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def build_date_range(config) -> pd.DatetimeIndex:
    w = config["windows"]
    return pd.date_range(start=w["date_start"], end=w["date_end"], freq="D")


@dataclass(frozen=True)
class WindowSpec:
    seq_length: int
    forecast_horizon: int
    stride: int

    @property
    def window_len(self) -> int:
        return self.seq_length + self.forecast_horizon

    @classmethod
    def from_config(cls, config) -> "WindowSpec":
        w = config["windows"]
        return cls(
            seq_length=int(w["seq_length"]),
            forecast_horizon=int(w["forecast_horizon"]),
            stride=int(w["stride"]),
        )


def generate_windows(n_timesteps: int, spec: WindowSpec) -> list[tuple[int, int]]:
    """Return [(start_idx, end_idx), ...] with end_idx exclusive."""
    wl = spec.window_len
    max_start = max(n_timesteps - wl + 1, 0)
    return [(s, s + wl) for s in range(0, max_start, spec.stride)]


def window_table(date_range: pd.DatetimeIndex, spec: WindowSpec) -> pd.DataFrame:
    """Build a per-window table with start/day3 dates (day3 = last window day)."""
    windows = generate_windows(len(date_range), spec)
    rows = []
    for idx, (start_idx, end_idx) in enumerate(windows):
        rows.append(
            {
                "window_index": idx,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_date": date_range[start_idx],
                "day3_date": date_range[end_idx - 1],
            }
        )
    return pd.DataFrame(rows)
