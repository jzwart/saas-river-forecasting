"""make_splits.py — deterministic temporal 80/20 split (plan §3, §6.4).

Decision (§2.2 + §3): a single global temporal cutoff placed inside the 2020 dry
season. The cutoff is the ``split.wetdry_quantile`` quantile of the wet/dry
(HoboWetDry0.05) label dates, so the scarce classification signal splits ~80/20:

    train = windows whose day-3 (forecast) date <= cutoff
    val   = windows whose day-3 date  > cutoff

Discharge keeps its full 1980-2020 history in train (train/val discharge volumes
are deliberately unbalanced — an accepted consequence of an honest temporal split).

Outputs (new filenames, never clobbering the released window_split_map.csv):
    data/retrain/window_split_map_temporal80.csv   window_index, start_date, day3_date, split
    data/retrain/split_meta.json                    cutoff + rule + per-split coverage

Run:  uv run python -m rgcn.pipeline.make_splits
"""

from __future__ import annotations

import json

import pandas as pd

from .config import load_config
from .windows import WindowSpec, build_date_range, window_table


def compute_cutoff(obs_csv, quantile: float) -> pd.Timestamp:
    """Cutoff = the given quantile of wet/dry label dates."""
    wd = pd.read_csv(obs_csv, usecols=["Date", "HoboWetDry0.05"])
    wd["Date"] = pd.to_datetime(wd["Date"])
    label_dates = wd.loc[wd["HoboWetDry0.05"].notna(), "Date"]
    if label_dates.empty:
        raise RuntimeError("No wet/dry labels found in obs.csv; cannot place cutoff.")
    # quantile(interpolation='lower') keeps the cutoff on an actual label date.
    return pd.Timestamp(label_dates.quantile(quantile, interpolation="lower")).normalize()


def _coverage(obs_csv, cutoff: pd.Timestamp, date_range: pd.DatetimeIndex) -> dict:
    """Per-split observation coverage, reported by observation date vs cutoff.

    Clipped to the modeled date range (obs.csv extends to 2023 but drivers and
    windows stop at date_end, so later observations are never modeled)."""
    obs = pd.read_csv(
        obs_csv, usecols=["Date", "HoboWetDry0.05", "Discharge_CMS"]
    )
    obs["Date"] = pd.to_datetime(obs["Date"])
    obs = obs[(obs["Date"] >= date_range[0]) & (obs["Date"] <= date_range[-1])]
    is_train = obs["Date"] <= cutoff

    def block(mask):
        sub = obs.loc[mask]
        wd = sub["HoboWetDry0.05"].dropna()
        return {
            "discharge_obs": int(sub["Discharge_CMS"].notna().sum()),
            "wetdry_obs": int(wd.shape[0]),
            "wetdry_wet": int((wd == 1.0).sum()),
            "wetdry_dry": int((wd == 0.0).sum()),
            "date_min": (sub["Date"].min().date().isoformat() if not sub.empty else None),
            "date_max": (sub["Date"].max().date().isoformat() if not sub.empty else None),
        }

    return {"train": block(is_train), "val": block(~is_train)}


def main() -> int:
    config = load_config()
    spec = WindowSpec.from_config(config)
    date_range = build_date_range(config)

    quantile = float(config["split"]["wetdry_quantile"])
    obs_csv = config.path("obs_csv")
    cutoff = compute_cutoff(obs_csv, quantile)

    table = window_table(date_range, spec)
    table["split"] = table["day3_date"].apply(
        lambda d: "train" if d <= cutoff else "val"
    )

    out = table[["window_index", "start_date", "day3_date", "split"]].copy()
    out["start_date"] = out["start_date"].dt.date.astype(str)
    out["day3_date"] = out["day3_date"].dt.date.astype(str)

    split_map_path = config.path("split_map")
    split_map_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(split_map_path, index=False)

    counts = out["split"].value_counts().to_dict()
    coverage = _coverage(obs_csv, cutoff, date_range)
    meta = {
        "rule": (
            "Single global temporal cutoff = quantile(wetdry_label_dates, q). "
            "Window is train if its day-3 (last) date <= cutoff, else val."
        ),
        "wetdry_quantile": quantile,
        "cutoff_date": cutoff.date().isoformat(),
        "date_range": [str(date_range[0].date()), str(date_range[-1].date())],
        "window": {
            "seq_length": spec.seq_length,
            "forecast_horizon": spec.forecast_horizon,
            "stride": spec.stride,
            "window_len": spec.window_len,
        },
        "n_windows": int(len(out)),
        "window_counts": {k: int(v) for k, v in counts.items()},
        "coverage_by_obs_date": coverage,
    }
    meta_path = config.path("split_meta")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Cutoff date: {meta['cutoff_date']} (q={quantile} of wet/dry label dates)")
    print(f"Windows: {counts.get('train', 0):,} train / {counts.get('val', 0):,} val "
          f"({len(out):,} total)")
    print("Wet/dry obs (by date):")
    for split in ("train", "val"):
        c = coverage[split]
        print(f"  {split}: wet={c['wetdry_wet']} dry={c['wetdry_dry']} "
              f"(discharge={c['discharge_obs']:,})  {c['date_min']}..{c['date_max']}")
    print(f"Wrote {split_map_path}")
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
