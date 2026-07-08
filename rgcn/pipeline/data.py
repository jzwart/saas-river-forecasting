"""Shared data layer: load the monolithic frames, build the feature/target grids,
compute train-only normalization, and cache the result.

Replaces the missing per-node CSV directories (the released dataset globbed
per-node files) by slicing the monolithic met_drivers.csv / obs.csv / static_vars.csv
directly and gridding them into dense arrays.

Grids (T = len(date_range), N = len(node_ids)):
    X_time   (T, N, 20)  drivers + obs-lags + MaxDepth + month/day  (FEATURE_VARS[:20])
    X_static (N, 17)     normalized static watershed vars           (FEATURE_VARS[20:])
    y_all    (T, N, 2)   [HoboWetDry0.05, log1p(Discharge_CMS)]

Normalization (plan §6.2): z-score with statistics computed on TRAIN dates only
(date <= split cutoff), saved to feature_scaler.json for reproducible inference.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import features as F

DRY_THRESHOLD = 0.00014  # Discharge_CMS <= this => impute HoboWetDry0.05 = 0 (dry)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def cache_drivers_parquet(config, force: bool = False):
    """Convert the 2.13 GB met_drivers.csv to parquet once for fast reloads."""
    pq = config.path("drivers_parquet")
    if pq.exists() and not force:
        return pq
    pq.parent.mkdir(parents=True, exist_ok=True)
    print(f"Caching drivers to parquet: {pq} (one-time)")
    df = pd.read_csv(config.path("met_drivers_csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    df.to_parquet(pq, index=False)
    return pq


def load_drivers(config) -> pd.DataFrame:
    pq = cache_drivers_parquet(config)
    return pd.read_parquet(pq)


def load_obs(config, impute_dry: bool = True) -> pd.DataFrame:
    """Load obs.csv. Optionally impute HoboWetDry0.05 = dry where discharge is
    near-zero (released behavior)."""
    cols = [
        "NHDPlusID", "Date", "Discharge_CMS", "HoboWetDry0.05",
        "MaxDepth_cm", "MaxDepth_Threshold", "MaxDepth_Censor",
    ]
    df = pd.read_csv(config.path("obs_csv"), usecols=cols)
    df["Date"] = pd.to_datetime(df["Date"])
    if impute_dry:
        dry = df["Discharge_CMS"].notna() & (df["Discharge_CMS"] <= DRY_THRESHOLD)
        df.loc[dry, "HoboWetDry0.05"] = 0.0
    return df


def load_statics(config) -> pd.DataFrame:
    df = pd.read_csv(config.path("static_vars_csv"))
    keep = ["NHDPlusID"] + F.STATIC_VARS
    return df[keep].copy()


def get_node_ids(config) -> list[int]:
    """Canonical node ordering = sorted NHDPlusID over the static-vars table
    (all 793 nodes; matches sorted(graph.nodes()))."""
    return sorted(int(x) for x in load_statics(config)["NHDPlusID"].unique())


def read_split_meta(config) -> dict:
    """split_meta.json (run make_splits first)."""
    meta_path = config.path("split_meta")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Run: uv run python -m rgcn.pipeline.make_splits"
        )
    with open(meta_path) as fh:
        return json.load(fh)


def train_date_mask(config, date_range: pd.DatetimeIndex) -> tuple[np.ndarray, str]:
    """Boolean (T,) mask of dates whose data may inform normalization stats,
    derived from the split rule in split_meta.json.

    - cutoff rule (quantile split): train dates = date <= cutoff.
    - exclude_blocks rule (holdout-blocks split): train dates = everything
      outside the guarded holdout blocks.
    Returns (mask, description) — the description is stored in the scaler json.
    """
    meta = read_split_meta(config)
    rule = meta.get("train_date_rule") or {"type": "cutoff", "cutoff": meta["cutoff_date"]}
    if rule["type"] == "cutoff":
        cutoff = pd.Timestamp(rule["cutoff"])
        return np.asarray(date_range <= cutoff), f"date <= {cutoff.date()}"
    if rule["type"] == "exclude_blocks":
        mask = np.ones(len(date_range), dtype=bool)
        for s, e in rule["blocks_with_guard"]:
            mask &= ~((date_range >= pd.Timestamp(s)) & (date_range <= pd.Timestamp(e)))
        desc = "exclude " + ", ".join(f"{s}..{e}" for s, e in rule["blocks_with_guard"])
        return mask, desc
    raise ValueError(f"Unknown train_date_rule type: {rule['type']}")


# --------------------------------------------------------------------------- #
# Grid construction
# --------------------------------------------------------------------------- #
def _grid_from_long(long_df, value_col, t_idx, n_idx, shape) -> np.ndarray:
    """Scatter a long-format column onto a dense (T, N) grid (NaN elsewhere)."""
    grid = np.full(shape, np.nan, dtype=np.float64)
    vals = long_df[value_col].to_numpy()
    present = ~np.isnan(vals)
    grid[t_idx[present], n_idx[present]] = vals[present]
    return grid


def _shift_time(grid: np.ndarray, lag: int) -> np.ndarray:
    """Shift a (T, N) grid down along the time axis by `lag` days (calendar lag);
    rows without history become NaN."""
    out = np.full_like(grid, np.nan)
    if lag < grid.shape[0]:
        out[lag:] = grid[:-lag]
    return out


def _zstats_train(grid: np.ndarray, train_mask_t: np.ndarray) -> tuple[float, float]:
    """Mean/std over train rows (date <= cutoff), ignoring NaN."""
    block = grid[train_mask_t]
    vals = block[~np.isnan(block)]
    if vals.size == 0:
        return 0.0, 1.0
    mean = float(vals.mean())
    std = float(vals.std())
    return mean, (std if std > 1e-12 else 1.0)


def build_arrays(config, impute_dry: bool = True):
    """Build X_time, X_static, y_all and the train-only scaler.

    Returns dict with arrays (float32), node_ids, dates (ISO strings), scaler.
    """
    from .windows import build_date_range

    date_range = build_date_range(config)
    node_ids = get_node_ids(config)
    n_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    T, N = len(date_range), len(node_ids)
    train_mask_t, train_rule = train_date_mask(config, date_range)  # (T,) boolean

    date_to_t = pd.Series(np.arange(T), index=date_range)

    scaler: dict[str, dict[str, float]] = {}

    # ---- time-varying feature block (T, N, 20) in FEATURE_VARS[:20] order ----
    time_feats = (
        F.DRIVER_VARS + F.DISCHARGE_LAG_VARS + F.WETDRY_LAG_VARS
        + F.MAXDEPTH_BINARY_VARS + F.MAXDEPTH_CONT_VARS + F.TEMPORAL_VARS
    )
    assert time_feats == F.FEATURE_VARS[:20]
    X_time = np.zeros((T, N, len(time_feats)), dtype=np.float32)
    col = {name: i for i, name in enumerate(time_feats)}

    # Drivers -------------------------------------------------------------
    drivers = load_drivers(config)
    drivers = drivers[drivers["Date"].isin(date_to_t.index)]
    dt = date_to_t.loc[drivers["Date"]].to_numpy()
    dn = drivers["NHDPlusID"].map(n_to_idx).to_numpy()
    keep = ~np.isnan(dn.astype(float))
    dt, dn = dt[keep], dn[keep].astype(int)
    for var in F.DRIVER_VARS:
        grid = np.full((T, N), np.nan, dtype=np.float64)
        grid[dt, dn] = drivers.loc[keep, var].to_numpy()
        mean, std = _zstats_train(grid, train_mask_t)
        scaler[var] = {"mean": mean, "std": std, "transform": "zscore"}
        X_time[:, :, col[var]] = np.nan_to_num((grid - mean) / std, nan=0.0)
    del drivers

    # Observations (targets, lags, MaxDepth) ------------------------------
    obs = load_obs(config, impute_dry=impute_dry)
    obs = obs[obs["Date"].isin(date_to_t.index)]
    ot = date_to_t.loc[obs["Date"]].to_numpy()
    on = obs["NHDPlusID"].map(n_to_idx).to_numpy()
    okeep = ~np.isnan(on.astype(float))
    ot, on = ot[okeep], on[okeep].astype(int)
    obs = obs.loc[okeep]

    def grid_of(colname):
        g = np.full((T, N), np.nan, dtype=np.float64)
        g[ot, on] = obs[colname].to_numpy()
        return g

    discharge = grid_of("Discharge_CMS")
    wetdry = grid_of("HoboWetDry0.05")

    # Targets: y[...,0]=wetdry, y[...,1]=log1p(discharge)
    y_all = np.full((T, N, len(F.TARGET_VARS)), np.nan, dtype=np.float32)
    y_all[:, :, F.WETDRY_IDX] = wetdry
    y_all[:, :, F.DISCHARGE_IDX] = np.log1p(np.clip(discharge, 0.0, None))

    # Discharge log-lags (log1p then z-score, train-only)
    for var, lag in zip(F.DISCHARGE_LAG_VARS, (1, 7)):
        lagged = _shift_time(discharge, lag)
        lagged = np.log1p(np.clip(lagged, 0.0, None))  # NaN stays NaN
        mean, std = _zstats_train(lagged, train_mask_t)
        scaler[var] = {"mean": mean, "std": std, "transform": "log1p+zscore"}
        X_time[:, :, col[var]] = np.nan_to_num((lagged - mean) / std, nan=0.0)

    # Wet/dry lags (binary 0/1, missing -> 0 dry)
    for var, lag in zip(F.WETDRY_LAG_VARS, (1, 7)):
        lagged = _shift_time(wetdry, lag)
        scaler[var] = {"transform": "none(binary)"}
        X_time[:, :, col[var]] = np.nan_to_num(lagged, nan=0.0)

    # MaxDepth binary flags (missing -> 0)
    for var in F.MAXDEPTH_BINARY_VARS:
        g = grid_of(var)
        scaler[var] = {"transform": "none(binary)"}
        X_time[:, :, col[var]] = np.nan_to_num(g, nan=0.0)

    # MaxDepth_cm (continuous, z-score train-only)
    for var in F.MAXDEPTH_CONT_VARS:
        g = grid_of(var)
        mean, std = _zstats_train(g, train_mask_t)
        scaler[var] = {"mean": mean, "std": std, "transform": "zscore"}
        X_time[:, :, col[var]] = np.nan_to_num((g - mean) / std, nan=0.0)
    del obs

    # Temporal month/day (z-score train-only so raw 1-31 magnitude can't
    # dominate the weather signal — plan §10)
    month = np.repeat(date_range.month.to_numpy()[:, None], N, axis=1).astype(np.float64)
    day = np.repeat(date_range.day.to_numpy()[:, None], N, axis=1).astype(np.float64)
    for var, grid in (("month", month), ("day", day)):
        mean, std = _zstats_train(grid, train_mask_t)
        scaler[var] = {"mean": mean, "std": std, "transform": "zscore"}
        X_time[:, :, col[var]] = ((grid - mean) / std).astype(np.float32)

    # ---- static block (N, 17), z-score over all nodes (all present in train) --
    statics = load_statics(config).set_index("NHDPlusID").reindex(node_ids)
    X_static = np.zeros((N, len(F.STATIC_VARS)), dtype=np.float32)
    for i, var in enumerate(F.STATIC_VARS):
        vals = statics[var].to_numpy(dtype=np.float64)
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals))
        std = std if std > 1e-12 else 1.0
        scaler[var] = {"mean": mean, "std": std, "transform": "zscore(static)"}
        X_static[:, i] = np.nan_to_num((vals - mean) / std, nan=0.0)

    return {
        "X_time": X_time,
        "X_static": X_static,
        "y_all": y_all,
        "node_ids": np.array(node_ids, dtype=np.int64),
        "dates": np.array([d.isoformat() for d in date_range]),
        "scaler": scaler,
        "train_rule": train_rule,
    }


def _arrays_path(config):
    if "feature_arrays" in config["paths"]:
        return config.path("feature_arrays")
    return config.path("drivers_parquet").parent / "feature_arrays.npz"


def save_arrays(config, arrays: dict):
    npz_path = _arrays_path(config)
    np.savez_compressed(
        npz_path,
        X_time=arrays["X_time"],
        X_static=arrays["X_static"],
        y_all=arrays["y_all"],
        node_ids=arrays["node_ids"],
        dates=arrays["dates"],
    )
    with open(config.path("scaler_json"), "w") as fh:
        json.dump({"train_dates": arrays["train_rule"], "features": arrays["scaler"]},
                  fh, indent=2)
    return npz_path


def load_arrays(config):
    npz_path = _arrays_path(config)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Run: uv run python -m rgcn.pipeline.prepare_data"
        )
    return np.load(npz_path, allow_pickle=False)
