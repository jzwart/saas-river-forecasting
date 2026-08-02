# RGCN retrain pipeline

Reproducible retraining of the RGCN wet/dry + discharge forecaster, fixing
three defects in the originally released model (unused meteorological drivers,
unfed static features, train/val normalization leakage) and replacing the
leaky interleaved evaluation split with honest temporal splits. See
`context/RGCN_RETRAIN_PLAN.md` for the full rationale and
`results/rgcn_eval_retrain*.md` for the resulting metrics.

## Requirements

- Linux machine with an NVIDIA GPU (training checks for CUDA and will refuse
  to run without it; ~13 GB VRAM at the default batch size 128).
  Driver must support CUDA 12.6 binaries (12.4+ drivers work via CUDA
  minor-version compatibility; the lockfile pins torch to the cu126 build on
  Linux).
- [uv](https://docs.astral.sh/uv/) (all commands below are `uv run`, which
  creates/uses the locked environment automatically).
- ~6 GB disk for raw data + ~3 GB for derived arrays/predictions.

## 1. Get the data

```bash
uv run python download_data.py
```

This fetches `obs.csv` and `static_vars.csv` from ScienceBase and the released
artifacts (graph, stream-order table, released model, split map) from Hugging
Face into `data/`. Downloaded ScienceBase files are verified against the
catalog MD5 checksums.

Some files may need a manual browser step (the downloader prints exact
instructions, destination paths, and expected MD5s for anything it cannot
fetch):

- **`met_drivers.csv` (2.13 GB)** is a request-gated S3 file on ScienceBase
  (CAPTCHA + async bundling). Request it via the printed URL and place it at
  `data/sciencebase/met_drivers.csv`.
- ScienceBase intermittently disables direct downloads on the (currently
  unpublished) data-release item — as of Aug 2026 `obs.csv` 404s and must be
  downloaded from the item page manually. This should resolve once the
  Zwart et al. (2026) data release is published.

## 2. Run the pipeline

Each stage reads `rgcn/config.yml` by default; set `RGCN_CONFIG` to a variant
config to run an alternative split end to end (all outputs are
variant-specific filenames, nothing is overwritten):

```bash
# 1. Temporal train/val split (writes split map + split_meta.json)
uv run python -m rgcn.pipeline.make_splits

# 2. Feature/target arrays with train-only normalization (~5 min; caches the
#    2.13 GB driver CSV to parquet on first run)
uv run python -m rgcn.pipeline.prepare_data

# 3. River-network topology graph (793 nodes / 780 edges, rebuilt from
#    static_vars.csv; cross-checks against the released graph if present)
uv run python -m rgcn.pipeline.build_graph

# 4. Train (~12 s/epoch on an RTX 6000 Ada; ~20 min with early stopping)
CUDA_VISIBLE_DEVICES=0 uv run python -m rgcn.pipeline.train

# 5. Export per-day prediction CSVs for all windows
CUDA_VISIBLE_DEVICES=0 uv run python -m rgcn.pipeline.export_predictions

# 6. Metrics report (writes results/rgcn_eval_retrain*.md)
uv run python -m rgcn.pipeline.eval_report
```

`uv run python -m rgcn.pipeline.train --smoke` runs a 2-epoch sanity check on
a 400-window subset (writes a separate `*_smoke.pt` checkpoint).

To verify the defect fixes on a trained checkpoint:

```bash
uv run python rgcn/inspect_driver_weights.py   # driver weight magnitudes
```

## Split variants

| Config | Split | Val wet/dry labels |
|---|---|--:|
| `rgcn/config.yml` | temporal cutoff @ q0.80 of label dates (2020-09-28) | 576 |
| `rgcn/config_q65.yml` | temporal cutoff @ q0.65 (2020-09-10) | 968 |
| `rgcn/config_phases.yml` | 3×12-day blocked holdout (drying / peak-dry / rewetting) with 7-day guard buffers | 784 |

Example: `RGCN_CONFIG=rgcn/config_phases.yml uv run python -m rgcn.pipeline.make_splits`
(then stages 2–6 with the same env var; stage 3 only needs to run once).

All runs are seeded (seed 42, cuDNN deterministic). Exact loss values can
still vary in the last digits across GPU models/driver versions; the reported
metrics reproduce to ~3 decimal places on same-generation hardware.

## Layout

```
rgcn/pipeline/
├── config.py        config loading; RGCN_CONFIG env var; repo-relative paths
├── features.py      canonical feature list (11 drivers + 4 obs-lags + 3 depth
│                    + month/day + 17 statics = 37); order matters
├── windows.py       31-day window grid (28 history + 3 forecast, stride 3)
├── make_splits.py   quantile_cutoff | holdout_blocks split methods
├── data.py          dense (T,N,F) grids; train-only z-score; dry imputation
├── prepare_data.py  builds + caches feature_arrays.npz / feature_scaler.json
├── build_graph.py   topology from static_vars FromNode/ToNode
├── model.py         RGCN_v2 (LSTM + graph conv), batched (B,N,T,F) forward
├── dataset.py       window dataset + split-index loading
├── losses.py        masked RMSE + weighted-BCE multitask loss
├── train.py         batched training loop, early stopping, checkpointing
├── export_predictions.py  per-horizon prediction CSVs
└── eval_report.py   classification + regression metrics report
```

Golden rule: released artifacts under `data/huggingface/` are never
overwritten — every pipeline output uses a new filename under `data/retrain/`.
