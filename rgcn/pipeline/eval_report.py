"""eval_report.py — metrics report on the retrain predictions (plan step 11 / §9).

Reproduces rgcn_eval.ipynb's classification analysis (accuracy / ROC-AUC / F1 by
horizon x stream-order group, per-order breakdown, confusion matrices) on the
val split, and adds discharge regression metrics (NSE / KGE / RMSE / MAPE) now
that discharge is genuinely modeled. Parameterized (no hardcoded release paths)
so it can also be pointed at the released predictions for comparison.

Run:  uv run python -m rgcn.pipeline.eval_report
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from .config import load_config


def _load(config):
    pred_dir = config.path("predictions_dir")
    split = pd.read_csv(config.path("split_map"))[["window_index", "split"]]
    order = pd.read_csv(
        config.path("released_graph").parent / "nhd_id_stream_order_permanence.csv",
        usecols=["NHDPlusID", "StreamOrde"],
    )
    dfs = {}
    for h in (1, 2, 3):
        df = pd.read_csv(pred_dir / f"train_val_predictions_day{h}.csv")
        df = df.merge(split, on="window_index", how="left")
        df = df.merge(order, left_on="site_id", right_on="NHDPlusID", how="left")
        dfs[f"Day {h}"] = df
    return dfs


def _cls_metrics(y_true, y_pred_label, y_pred_prob):
    acc = accuracy_score(y_true, y_pred_label)
    f1 = f1_score(y_true, y_pred_label, zero_division=0)
    roc = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return acc, roc, f1


def _reg_metrics(true_lin, pred_lin):
    """NSE / KGE / RMSE / MAPE in linear discharge space."""
    m = ~np.isnan(true_lin) & ~np.isnan(pred_lin)
    t, p = true_lin[m], pred_lin[m]
    if len(t) < 2:
        return dict(N=int(len(t)), NSE=np.nan, KGE=np.nan, RMSE=np.nan, MAPE=np.nan)
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    nse = float(1 - np.sum((t - p) ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-10))
    r = float(np.corrcoef(t, p)[0, 1])
    alpha = float(p.std() / (t.std() + 1e-10))
    beta = float(p.mean() / (t.mean() + 1e-10))
    kge = float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    nz = np.abs(t) > 1e-8
    mape = float(np.mean(np.abs((t[nz] - p[nz]) / t[nz])) * 100) if nz.any() else np.nan
    return dict(N=int(len(t)), NSE=nse, KGE=kge, RMSE=rmse, MAPE=mape)


GROUPS = {
    "All": lambda d: d,
    "Headwaters (<=2)": lambda d: d[d["StreamOrde"] <= 2],
    "Tailwaters (>=3)": lambda d: d[d["StreamOrde"] >= 3],
}


def main() -> int:
    config = load_config()
    dfs = _load(config)
    # Validation, wet/dry-labeled rows per horizon + aggregate.
    val = {name: d[(d["has_true_label"]) & (d["split"] == "val")].copy()
           for name, d in dfs.items()}
    val["All Horizons"] = pd.concat(val.values(), ignore_index=True)

    lines = ["# RGCN retrain — evaluation report", ""]
    lines.append(f"Checkpoint: `{config['paths']['checkpoint']}` | "
                 f"split: `{config.path('split_map').name}`")
    lines.append("")

    # ---- Classification: horizon x group ----
    lines.append("## Wet/dry classification (val split)")
    lines.append("")
    lines.append("| Horizon | Group | N | Accuracy | ROC-AUC | F1 |")
    lines.append("|---|---|--:|--:|--:|--:|")
    for hname, d in val.items():
        for gname, gf in GROUPS.items():
            s = gf(d)
            if len(s) == 0:
                continue
            yt = s["true_wetdry"].round().astype(int).values
            acc, roc, f1 = _cls_metrics(yt, s["pred_wetdry_label"].values,
                                        s["pred_wetdry_prob"].values)
            lines.append(f"| {hname} | {gname} | {len(s):,} | {acc:.3f} | "
                         f"{roc:.3f} | {f1:.3f} |")
    lines.append("")

    # ---- Stream-order breakdown (All Horizons) ----
    agg = val["All Horizons"]
    lines.append("## Stream-order breakdown (All Horizons, val)")
    lines.append("")
    lines.append("| Order | N | Accuracy | ROC-AUC | F1 |")
    lines.append("|--:|--:|--:|--:|--:|")
    for order in sorted(agg["StreamOrde"].dropna().unique().astype(int)):
        s = agg[agg["StreamOrde"] == order]
        yt = s["true_wetdry"].round().astype(int).values
        acc, roc, f1 = _cls_metrics(yt, s["pred_wetdry_label"].values,
                                    s["pred_wetdry_prob"].values)
        lines.append(f"| {order} | {len(s):,} | {acc:.3f} | {roc:.3f} | {f1:.3f} |")
    lines.append("")

    # ---- Confusion matrix (All Horizons) ----
    yt = agg["true_wetdry"].round().astype(int).values
    yp = agg["pred_wetdry_label"].values
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    lines.append("## Confusion matrix (All Horizons, val)")
    lines.append("")
    lines.append("| Observed \\ Pred | Dry | Wet |")
    lines.append("|---|--:|--:|")
    lines.append(f"| **Dry** | {cm[0,0]:,} | {cm[0,1]:,} |")
    lines.append(f"| **Wet** | {cm[1,0]:,} | {cm[1,1]:,} |")
    lines.append("")

    # ---- Discharge regression (val) ----
    lines.append("## Discharge regression (val split, linear CMS)")
    lines.append("")
    lines.append("| Horizon | N | NSE | KGE | RMSE | MAPE% |")
    lines.append("|---|--:|--:|--:|--:|--:|")
    for hname, d in dfs.items():
        s = d[d["split"] == "val"]
        m = _reg_metrics(s["true_discharge"].values, s["pred_discharge"].values)
        lines.append(f"| {hname} | {m['N']:,} | {m['NSE']:.3f} | {m['KGE']:.3f} | "
                     f"{m['RMSE']:.4f} | {m['MAPE']:.1f} |")
    lines.append("")

    # Committable (not under gitignored data/); per-variant via paths.eval_report.
    out = config.repo_root / config["paths"].get("eval_report", "results/rgcn_eval_retrain.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
