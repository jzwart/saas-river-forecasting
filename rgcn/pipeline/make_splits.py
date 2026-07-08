"""make_splits.py — deterministic temporal splits (plan §3, §6.4).

Two methods, selected by ``split.method``:

quantile_cutoff (default; decision §2.2 + §3): a single global temporal cutoff
placed inside the 2020 dry season. The cutoff is the ``split.wetdry_quantile``
quantile of the wet/dry (HoboWetDry0.05) label dates:

    train = windows whose day-3 (forecast) date <= cutoff
    val   = windows whose day-3 date  > cutoff

holdout_blocks: multi-phase blocked holdout. All real wet/dry labels sit in one
season (Jun-Oct 2020), so year-level holdout is impossible; instead val = windows
whose forecast (loss) days fall entirely inside one of the ``split.holdout_blocks``
date ranges (chosen to span distinct hydrologic phases: drying limb, peak dry,
rewetting). Windows whose loss days land within ``split.guard_days`` of a block
are marked ``buffer`` and used by neither side (guards against day-to-day
autocorrelation leaking adjacent training targets into val).

Discharge keeps its full 1980-2020 history in train (train/val discharge volumes
are deliberately unbalanced — an accepted consequence of an honest temporal split).

Outputs (new filenames, never clobbering the released window_split_map.csv):
    <paths.split_map>    window_index, start_date, day3_date, split
    <paths.split_meta>   rule + per-split coverage (drives normalization masks)

Run:  [RGCN_CONFIG=rgcn/config_x.yml] uv run python -m rgcn.pipeline.make_splits
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


def _coverage(obs_csv, date_masks: dict, date_range: pd.DatetimeIndex) -> dict:
    """Per-split observation coverage. ``date_masks`` maps split name to a
    callable(date_series) -> boolean mask.

    Clipped to the modeled date range (obs.csv extends to 2023 but drivers and
    windows stop at date_end, so later observations are never modeled)."""
    obs = pd.read_csv(
        obs_csv, usecols=["Date", "HoboWetDry0.05", "Discharge_CMS"]
    )
    obs["Date"] = pd.to_datetime(obs["Date"])
    obs = obs[(obs["Date"] >= date_range[0]) & (obs["Date"] <= date_range[-1])]

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

    return {name: block(fn(obs["Date"])) for name, fn in date_masks.items()}


def _in_any(dates: pd.Series, blocks: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    mask = pd.Series(False, index=dates.index)
    for s, e in blocks:
        mask |= (dates >= s) & (dates <= e)
    return mask


def _split_quantile_cutoff(config, table, date_range):
    """train = day-3 date <= cutoff, val = after. Returns (table, meta-fragment,
    coverage date-mask callables)."""
    quantile = float(config["split"]["wetdry_quantile"])
    cutoff = compute_cutoff(config.path("obs_csv"), quantile)
    table["split"] = table["day3_date"].apply(
        lambda d: "train" if d <= cutoff else "val"
    )
    meta = {
        "rule": (
            "Single global temporal cutoff = quantile(wetdry_label_dates, q). "
            "Window is train if its day-3 (last) date <= cutoff, else val."
        ),
        "wetdry_quantile": quantile,
        "cutoff_date": cutoff.date().isoformat(),
        "train_date_rule": {"type": "cutoff", "cutoff": cutoff.date().isoformat()},
    }
    masks = {"train": lambda d: d <= cutoff, "val": lambda d: d > cutoff}
    print(f"Cutoff date: {meta['cutoff_date']} (q={quantile} of wet/dry label dates)")
    return table, meta, masks


def _split_holdout_blocks(config, table, date_range):
    """val = windows whose forecast (loss) days fall entirely inside a holdout
    block; windows with loss days within guard_days of a block become buffer."""
    spec_horizon = int(config["windows"]["forecast_horizon"])
    guard = pd.Timedelta(days=int(config["split"]["guard_days"]))
    blocks = [
        (pd.Timestamp(s), pd.Timestamp(e)) for s, e in config["split"]["holdout_blocks"]
    ]
    guarded = [(s - guard, e + guard) for s, e in blocks]

    day3 = table["day3_date"]
    day1 = day3 - pd.Timedelta(days=spec_horizon - 1)  # first loss day
    is_val = pd.Series(False, index=table.index)
    for s, e in blocks:
        is_val |= (day1 >= s) & (day3 <= e)
    # Loss span touches a guarded block if it starts before the guard end and
    # ends after the guard start.
    touches = pd.Series(False, index=table.index)
    for s, e in guarded:
        touches |= (day1 <= e) & (day3 >= s)
    table["split"] = "train"
    table.loc[touches, "split"] = "buffer"
    table.loc[is_val, "split"] = "val"

    meta = {
        "rule": (
            "Multi-phase blocked holdout: val = windows whose forecast (loss) days "
            "fall entirely inside a holdout block; loss days within guard_days of a "
            "block => buffer (excluded from train to limit autocorrelation leakage). "
            "All real wet/dry labels are Jun-Oct 2020, so blocks sample distinct "
            "hydrologic phases of that season."
        ),
        "holdout_blocks": [[str(s.date()), str(e.date())] for s, e in blocks],
        "guard_days": int(config["split"]["guard_days"]),
        "train_date_rule": {
            "type": "exclude_blocks",
            "blocks": [[str(s.date()), str(e.date())] for s, e in blocks],
            "guard_days": int(config["split"]["guard_days"]),
            "blocks_with_guard": [[str(s.date()), str(e.date())] for s, e in guarded],
        },
    }
    masks = {
        "train": lambda d: ~_in_any(d, guarded),
        "val": lambda d: _in_any(d, blocks),
        "buffer": lambda d: _in_any(d, guarded) & ~_in_any(d, blocks),
    }
    print(f"Holdout blocks (guard={config['split']['guard_days']}d): "
          + ", ".join(f"{s.date()}..{e.date()}" for s, e in blocks))
    return table, meta, masks


def main() -> int:
    config = load_config()
    spec = WindowSpec.from_config(config)
    date_range = build_date_range(config)
    obs_csv = config.path("obs_csv")

    method = config["split"].get("method", "quantile_cutoff")
    table = window_table(date_range, spec)
    if method == "quantile_cutoff":
        table, meta, masks = _split_quantile_cutoff(config, table, date_range)
    elif method == "holdout_blocks":
        table, meta, masks = _split_holdout_blocks(config, table, date_range)
    else:
        raise ValueError(f"Unknown split.method: {method}")

    out = table[["window_index", "start_date", "day3_date", "split"]].copy()
    out["start_date"] = out["start_date"].dt.date.astype(str)
    out["day3_date"] = out["day3_date"].dt.date.astype(str)

    split_map_path = config.path("split_map")
    split_map_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(split_map_path, index=False)

    counts = out["split"].value_counts().to_dict()
    coverage = _coverage(obs_csv, masks, date_range)
    meta.update({
        "method": method,
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
    })
    meta_path = config.path("split_meta")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print("Windows: " + " / ".join(f"{counts.get(k, 0):,} {k}"
                                   for k in ("train", "val", "buffer") if k in counts)
          + f" ({len(out):,} total)")
    print("Wet/dry obs (by date):")
    for split, c in coverage.items():
        print(f"  {split}: wet={c['wetdry_wet']} dry={c['wetdry_dry']} "
              f"(discharge={c['discharge_obs']:,})  {c['date_min']}..{c['date_max']}")
    print(f"Wrote {split_map_path}")
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
