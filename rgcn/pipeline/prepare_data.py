"""prepare_data.py — build and cache the normalized feature/target arrays (plan §6.1).

Slices the monolithic met_drivers.csv / obs.csv / static_vars.csv into dense grids,
computes train-only normalization, and writes:
    data/retrain/feature_arrays.npz   X_time, X_static, y_all, node_ids, dates
    data/retrain/feature_scaler.json  per-feature normalization stats + cutoff

Prereq: run make_splits first (needs the cutoff date). Run:
    uv run python -m rgcn.pipeline.prepare_data
"""

from __future__ import annotations

import numpy as np

from .config import load_config
from .data import build_arrays, save_arrays


def main() -> int:
    config = load_config()
    print("Building feature/target arrays (train-only normalization)...")
    arrays = build_arrays(config, impute_dry=True)

    X_time, X_static, y = arrays["X_time"], arrays["X_static"], arrays["y_all"]
    print(f"  X_time  {X_time.shape}  {X_time.nbytes/1e9:.2f} GB")
    print(f"  X_static {X_static.shape}")
    print(f"  y_all   {y.shape}")
    print(f"  train-date rule (normalization): {arrays['train_rule']}")

    # Sanity: driver columns must be non-zero (the original bug was all-zero drivers)
    driver_block = X_time[:, :, :11]
    print(f"  driver block: mean|x|={np.abs(driver_block).mean():.4f} "
          f"nonzero_frac={np.mean(driver_block != 0):.3f}")
    wetdry_valid = np.isfinite(y[:, :, 0]) & ~np.isnan(y[:, :, 0])
    disch_valid = ~np.isnan(y[:, :, 1])
    print(f"  wet/dry target obs: {int(wetdry_valid.sum()):,}")
    print(f"  discharge target obs: {int(disch_valid.sum()):,}")

    npz_path = save_arrays(config, arrays)
    print(f"Wrote {npz_path}")
    print(f"Wrote {config.path('scaler_json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
