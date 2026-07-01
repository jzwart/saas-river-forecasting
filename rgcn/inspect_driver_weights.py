#!/usr/bin/env python3
"""
Inspect the RGCN input weights and show that the meteorological driver features
are effectively unused by the trained model.

The RGCN's input projection ``weight_ih`` has shape (20, 256): one row per input
feature, mapping into the four LSTM gates (4 * hidden_dim = 4 * 64 = 256). A
feature can only influence the model through its row of ``weight_ih``. If a
feature's row is ~0, that feature is ignored no matter what value it takes.

The 20 features are, in order:
    0-10 : 11 GridMET drivers            <-- expected to be ~0 (unused)
    11-14: Discharge/Hobo wet-dry lags   <-- active
    15-17: MaxDepth (censor/threshold/cm)<-- also ~0 (unused)
    18-19: month, day                    <-- active

Usage:
    python inspect_driver_weights.py
    python inspect_driver_weights.py --checkpoint /path/to/best_model.pt
    python inspect_driver_weights.py --threshold 1e-12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Feature order matches FullGraphTemporalDataset: drivers, obs lags, obs raw, temporal.
FEATURE_NAMES = [
    # 11 GridMET meteorological drivers
    "etalfalfa", "etgrass", "prcp", "rhmax", "rhmin", "sph", "srad",
    "tmax", "tmin", "vp", "ws",
    # observation-derived lag features
    "Discharge_CMS_lag_1", "Discharge_CMS_lag_7",
    "HoboWetDry0.05_lag_1", "HoboWetDry0.05_lag_7",
    # observation-derived raw features
    "MaxDepth_Censor", "MaxDepth_Threshold", "MaxDepth_cm",
    # temporal features
    "month", "day",
]

DRIVER_IDX = list(range(0, 11))          # the 11 GridMET drivers
ACTIVE_IDX = [11, 12, 13, 14, 18, 19]    # lags + month/day (known-used features)


def main() -> int:
    default_ckpt = Path(__file__).resolve().parent.parent / "data" / "huggingface" / "best_model.pt"

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_ckpt,
        help=f"Path to the RGCN checkpoint (default: {default_ckpt})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-8,
        help="A feature is flagged 'UNUSED' if the max |weight| in its row is below this (default: 1e-8).",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        parser.error(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Download it with:  python download_data.py"
        )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    if "weight_ih" not in state_dict:
        parser.error(
            "'weight_ih' not found in the checkpoint state_dict. "
            f"Available keys: {list(state_dict.keys())}"
        )

    W = state_dict["weight_ih"]  # (num_features, 4 * hidden_dim)
    n_features = W.shape[0]

    # Prefer feature names stored in the checkpoint (retrain checkpoints record the
    # full 37-feature layout: drivers, lags, MaxDepth, month/day, then statics).
    # Fall back to the hardcoded 20-feature released layout.
    feature_names = ckpt.get("feature_vars") if isinstance(ckpt, dict) else None
    if not feature_names:
        feature_names = FEATURE_NAMES
        if n_features != len(FEATURE_NAMES):
            print(
                f"WARNING: checkpoint has {n_features} input features but this script "
                f"expects {len(FEATURE_NAMES)} and stores no feature_vars. "
                f"Feature labels may be misaligned.\n"
            )

    # Drivers are always the first 11 rows; "active" reference = the known-used
    # obs lags + month/day, located by name when available.
    driver_idx = list(range(0, min(11, n_features)))
    if feature_names is FEATURE_NAMES:
        active_idx = ACTIVE_IDX
    else:
        known_active = {
            "Discharge_CMS_lag_1", "Discharge_CMS_lag_7",
            "HoboWetDry0.05_lag_1", "HoboWetDry0.05_lag_7", "month", "day",
        }
        active_idx = [i for i, n in enumerate(feature_names) if n in known_active]

    print("=" * 74)
    print(f"RGCN input-weight inspection  ({args.checkpoint})")
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    val_loss = ckpt.get("val_loss") if isinstance(ckpt, dict) else None
    if epoch is not None or val_loss is not None:
        print(f"checkpoint: epoch={epoch}  val_loss={val_loss}")
    print(f"weight_ih shape: {tuple(W.shape)}  (features x [4 gates * hidden])")
    print("=" * 74)
    print(f"{'idx':>3}  {'feature':22s} {'max|w|':>12s} {'mean|w|':>12s}   status")
    print("-" * 74)

    for i in range(n_features):
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        row = W[i]
        amax = row.abs().max().item()
        amean = row.abs().mean().item()
        status = "UNUSED (~0)" if amax < args.threshold else "active"
        marker = "  <== driver" if i in driver_idx else ""
        print(f"{i:3d}  {name:22s} {amax:12.3e} {amean:12.3e}   {status}{marker}")

    driver_max = W[driver_idx].abs().max().item()
    active_max = W[active_idx].abs().max().item()
    ratio = active_max / driver_max if driver_max > 0 else float("inf")

    print("=" * 74)
    print("SUMMARY")
    print("-" * 74)
    print(f"  Largest |weight| across the 11 drivers : {driver_max:.3e}")
    print(f"  Largest |weight| across active features: {active_max:.3e}")
    print(f"  Active features are ~{ratio:.1e}x larger than any driver weight.")
    drivers_unused = driver_max < args.threshold
    print()
    if drivers_unused:
        print("  RESULT: the meteorological driver weights are numerically zero.")
        print("          The trained model does NOT use the GridMET drivers.")
    else:
        print("  RESULT: at least one driver weight exceeds the threshold; drivers")
        print("          may be in use. Inspect the rows above.")
    print("=" * 74)

    return 0 if drivers_unused else 1


if __name__ == "__main__":
    raise SystemExit(main())
