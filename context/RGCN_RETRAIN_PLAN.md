# RGCN Retraining Plan — Session Handoff

**Purpose.** This is a self-contained handoff for retraining the RGCN from scratch on a GPU cluster, fixing several defects discovered in the released model/pipeline. It assumes the next session has **no memory of the prior investigation**, so it includes the findings, the decisions, and step-by-step instructions.

**Repo:** `saas-river-forecasting` (SAAS × USGS headwater streamflow, H.J. Andrews). Model of interest: `rgcn/` (RGCN_v2).

**Golden rule:** Do **not** overwrite or delete any currently released artifact. Everything current is a baseline to preserve (see §8). Work on a **new branch**.

---

## 1. Why we are retraining (findings from investigation)

Three defects were found in the released RGCN (`data/huggingface/best_model.pt`, `train_val_predictions_day{1,2,3}.csv`):

1. **The model does not use the meteorological driver features at all.**
   - `weight_ih` is `(20 features × 256 gates)`. The 11 GridMET driver rows have `max|w| ≈ 1e-26` (vs. ~0.3–0.7 for the live features). The 3 `MaxDepth` rows are likewise ~1e-26.
   - Only live inputs: `Discharge_CMS_lag_1/7`, `HoboWetDry0.05_lag_1/7`, `month`, `day` (+ graph topology via `A`).
   - The `1e-26` value is the signature of **weight decay shrinking a weight that received zero loss-gradient** — i.e., those input columns were **zero for every training sample**. Mechanism (which cell/step zeroed them) is unproven because the shipped notebooks were copied off a cluster and may be stale; the *outcome* is certain (perturbing drivers changes predictions by ≤1.2e-7).
   - A ready-made checker exists: **`rgcn/inspect_driver_weights.py`** (run it before and after retraining; after a correct retrain the driver rows should be non-negligible).

2. **Static watershed features are attached to the graph but never fed to the model.** `FullGraphTemporalDataset` only reads `gridmet_ts` and `obs_ts`; the 17 static NHDPlus vars are node attributes that never enter the input tensor.

3. **Normalization leaks validation data into training.** `build_graph.ipynb → normalize_continuous_columns` computes per-node z-score **over the full 1980–2020 series** (train+val together). Stats must be computed on the **training partition only**.

Additional reproducibility gaps (see §5): the split-generation script and the prediction-export script are **not in the repo**, per-node input CSV directories are missing, `gridmet_ts` is stripped from the released graph, and both notebooks hardcode a cluster path `/data/kripat/usgs-gnn`.

---

## 2. Decisions locked in for this retrain

1. **Include static features** as model inputs (broadcast each node's 17 static vars across all timesteps). ⚠️ *Caveat to keep in mind for the paper:* the draft's thesis is that the RGCN generalizes by using **topology instead of static site characteristics** (classical models fail via "static feature memorization"). Feeding statics reintroduces that risk — plan to show topology still matters and/or report with-vs-without statics.
2. **Split = temporal**, but generate a **NEW, reproducible 80/20 split** in-repo. Do **not** reuse the released `window_split_map.csv`.
3. **river-dl alignment = leakage fix only.** Skip pretraining, distance-weighted adjacency, Snakemake, and every other river-dl item. Keep the existing binary row-normalized downstream adjacency and the existing `rmse_masked` loss.
4. **Preserve all current results.** We will likely rewrite the paper's RGCN numbers, but every current artifact must be archived first (§8).

Also settled earlier: **site-based and random-window splits are out.** Site-based is architecturally impossible for the RGCN (full-graph message passing + lagged-obs inputs leak held-out nodes); random-window leaks temporally. Temporal is the honest, native choice. Spatial generalization stays assessed via **stream-order breakdown**, not a site holdout.

---

## 3. Critical data fact that shapes the split

**All wet/dry labels (`HoboWetDry0.05`) exist only in 2020-06-16 → 2020-10-29.** `Flow_Status` likewise 2020 only. `Discharge_CMS` spans 1980–2023.

Consequence: a naive temporal split over 1980–2020 would put **zero** wet/dry labels in validation. The new temporal 80/20 must be defined over **label-bearing windows** (dominantly the 2020 dry season for classification):

- Take every window that contains ≥1 valid label, sort chronologically by the window's forecast (day-3) date, assign the **earliest 80% → train, latest 20% → val**.
- This yields a meaningful test: the late-season val slice contains the dry-down, so it genuinely tests dry prediction.
- Document the exact rule + any seed in `make_splits.py` and commit the resulting `window_split_map.csv` alongside the script.

---

## 4. Cluster environment setup

Target hardware (confirmed): **10× NVIDIA RTX 6000 Ada (48 GB each)**, driver **550.107.02**, CUDA **12.4**. Compute capability **sm_89**.

### 4.1 Pick an idle GPU
At last check GPUs **2, 8, 9** were idle (~2 MiB); GPU 0 (the notebook's hardcoded `cuda:0`) was busy. Launch with e.g. `CUDA_VISIBLE_DEVICES=2` so the process maps to a free card. The model is tiny (~100k params) — a single GPU is ample; do **not** bother with multi-GPU/DDP.

### 4.2 Python env (conda)
```bash
conda create -n rgcn python=3.11 -y
conda activate rgcn
# CUDA build of torch matching driver 550 / CUDA 12.4 (cu121 also works)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas networkx pyyaml scipy scikit-learn matplotlib seaborn tqdm imbalanced-learn ipykernel jupyter
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Notes:
- **PyTorch Geometric is NOT required** — `RGCN_v2` is pure torch with a hand-rolled `A @ q_t`. Skip PyG (avoids CUDA-version pain). `CLAUDE.md` listing PyG is aspirational.
- `imbalanced-learn` (ADASYN) is only needed if you mirror the LSTM's resampling; the RGCN did **not** use ADASYN (kept for graph consistency). Include it only if you decide to.
- The eval notebook uses `scipy`, `scikit-learn`, `matplotlib`, `seaborn`.

### 4.3 Memory expectations
- `X_all` dense array ≈ (14,975 days × 793 nodes × ~37 feats) float32 ≈ **~1.6 GB CPU RAM** (grows from ~0.95 GB after adding statics). Fine.
- Peak GPU usage a few GB even at `batch_size=128`. 48 GB is overkill.

---

## 5. Data acquisition

From repo root:
```bash
python download_data.py --include drivers      # adds met_drivers.csv (2.13 GB) — REQUIRED for driver fix
# default set already includes obs.csv, static_vars.csv, graph, pivots, best_model.pt, splits
```
Sources: raw public data from ScienceBase (DOI 10.5066/P19R5TXW, item `6977e36dd4be02609dd04095`); derived artifacts from HF `michaeltm365/saas-river-forecasting`.

Key local files after download:
| File | Role |
|---|---|
| `data/sciencebase/obs.csv` | 159k obs: `HoboWetDry0.05`, `Discharge_CMS`, `Flow_Status`, `MaxDepth_*` (labels 2020 for wet/dry) |
| `data/sciencebase/met_drivers.csv` | 11.9M rows, 11 drivers, all 793 nodes, 1980–2020 (**raw**, unnormalized) |
| `data/sciencebase/static_vars.csv` | 17 static NHDPlus vars per node (also `static_vars_pivot.csv`) |
| `data/huggingface/hja_graph.gpickle` | topology + static attrs + `obs_ts`; **`gridmet_ts` is stripped (None)** |
| `data/huggingface/hja_edge_index.npz`, `degrees.parquet` | edges / in-out degree |
| `data/huggingface/nhd_id_stream_order_permanence.csv` | stream order (for stream-order eval) |
| `data/huggingface/best_model.pt` | released model (BASELINE — keep) |
| `data/huggingface/train_val_predictions_day{1,2,3}.csv` | released predictions (BASELINE — optional download `--include predictions`, 1.99 GB each) |

---

## 6. What to build (missing pieces) and what to fix

Convert the two notebooks into runnable, path-parameterized scripts (or papermill notebooks). Concrete work items:

### 6.1 `prepare_data.py` (new)
- Split `met_drivers.csv` and `obs.csv` by `NHDPlusID` into the per-node CSVs the dataset expects, **or** refactor `_load_gridmet_for_node`/`_load_obs_for_node` to slice the monolithic frames directly.

### 6.2 Fix `rgcn/build_graph.ipynb` (the driver + leakage fixes)
- Repoint `DATA_ROOT` from `/data/kripat/usgs-gnn` to a repo-relative/config path.
- **Attach gridmet:** the only `attach_time_series_to_graph(...)` call (cell ~22) uses `load_gridmet=False` ("# Already loaded") with no prior `load_gridmet=True`. Ensure drivers are actually attached and verify `graph.nodes[n]['gridmet_ts']` is a populated DataFrame for ~all nodes.
- **Leakage fix:** change `normalize_continuous_columns` to compute mean/std on the **training-partition dates only** (from §3's split), then apply to all dates. Recommended: **global per-feature** train-period stats (matches the LSTM's `StandardScaler`) rather than per-node; **save the scaler stats to disk** for reproducible inference. `month`/`day`, binary cols, and label cols remain excluded from normalization (consider z-scoring or sin/cos-encoding `month`/`day` since their raw magnitudes 1–31 currently dominate — optional).
- **Static features:** normalize the 17 static vars (train-only stats) and make them available to the dataset as per-node constants.

### 6.3 Fix `rgcn/train_gnn.ipynb`
- Repoint paths; keep `device` selection but honor `CUDA_VISIBLE_DEVICES` (don't hardcode `cuda:0` semantics beyond index 0).
- **Feed statics + drivers:** extend `FullGraphTemporalDataset` so each node's input vector = `[11 drivers, 4 obs lags, 3 MaxDepth, month, day, 17 statics broadcast across time]` → `input_dim ≈ 37`. Set `config["model"]["input_dim"] = len(FEATURE_VARS)` (cell 29 already does `len(FEATURE_VARS)`, so just grow the feature list).
- **Sync feature set with `lstm_all_sites`** where sensible: that notebook uses drivers + static + degrees + lags, `seq_len=30`, `StandardScaler` fit on train. Consider adding `in/out degree` to the RGCN feature set for parity (optional; topology already encodes some of this via `A`).
- Keep: `RGCN_v2`, binary row-normalized downstream `A` (`node_ids = sorted(graph.nodes())`), `rmse_masked`, multi-task heads, `hidden_dim=64`, `seq_length=28`, horizon 3.

### 6.4 `make_splits.py` (new — decision 2)
- Implement §3's temporal rule deterministically; commit both the script and the regenerated `window_split_map.csv` (as a **new** file, not overwriting the archived original — see §8).

### 6.5 `export_predictions.py` (new)
- Regenerate the `train_val_predictions_day{1,2,3}.csv` format that `rgcn/rgcn_eval.ipynb` consumes: columns `window_index, horizon_step, date, site_id, has_true_label, static__*, feature__*, true_*, pred_*`. `train_gnn.ipynb` cell 36 only prints metrics; this exporter must be written. Day-3 prediction = classification output at the **last timestep** of each 31-day window (window = 28 history + 3 horizon).

### 6.6 Config
- Add a single `config.yml` (or reuse `rgcn/rgcn_config.yaml`) with all paths + hyperparameters; remove hardcoded cluster paths everywhere.

---

## 7. Step-by-step execution order

1. `git checkout -b rgcn-retrain-<date>` (new branch).
2. **Archive baselines** (§8) — do this first, before any file is regenerated.
3. Create conda env (§4.2); pick idle GPU (§4.1).
4. `python download_data.py --include drivers`.
5. `prepare_data.py` → per-node inputs (or the slice-in-place refactor).
6. `make_splits.py` → new temporal 80/20 `window_split_map.csv` (new filename).
7. Fixed `build_graph` → new graph pickle **with** `gridmet_ts` attached, train-only normalization, statics normalized. Save as a **new** graph file (don't clobber the released `hja_graph.gpickle`).
8. Fixed `train_gnn` → train (drivers + statics + lags + temporal, input_dim ≈ 37). Save new checkpoint under a new name.
9. **Verify the fix:** `python rgcn/inspect_driver_weights.py --checkpoint <new_ckpt>` → driver rows should now be non-negligible (not ~1e-26). Also spot-check that `X_all[..., driver_cols]` is non-zero for weather-bearing nodes/dates *before* training (this is the check that would have caught the original bug).
10. `export_predictions.py` → new prediction CSVs.
11. Run `rgcn/rgcn_eval.ipynb` against the new predictions; regenerate metrics + the Day-3 dry-day map (that section already exists).
12. Then proceed to hyperparameter justification and loss-weight ablations (deferred items 5 & 6 from the plan; do after the corrected baseline is trained and verified).

---

## 8. Preserving current results (decision 4) — DO THIS FIRST

Before regenerating anything, archive the as-released state so the paper's current numbers remain reproducible:

- Create `results/as_released_2026-06/` (or a git tag `results-as-released`) and copy: `best_model.pt`, `train_val_predictions_day{1,2,3}.csv`, `window_split_map.csv`, `hja_graph.gpickle`, the current `rgcn_eval.ipynb` **with outputs**, and the `data/result_summaries/rgcn_eval_results.md`.
- Never overwrite these paths. All new artifacts get **new filenames** (e.g., `best_model_retrain.pt`, `window_split_map_temporal80.csv`, `hja_graph_drivers.gpickle`).
- Keep `rgcn/inspect_driver_weights.py` — it documents the original defect and validates the fix.
- Commit the archive on the new branch before step 7.

---

## 9. Acceptance criteria (how to know the retrain worked)

- `inspect_driver_weights.py` on the new checkpoint reports drivers as **active** (not `UNUSED (~0)`).
- Static-feature input rows in `weight_ih` are non-negligible.
- Normalization stats are computed from **train dates only** and saved to disk; re-running inference from the saved scaler reproduces predictions.
- The new `window_split_map.csv` is regenerable from `make_splits.py` (deterministic) and has labeled observations in **both** train and val (recall §3).
- `rgcn_eval.ipynb` runs end-to-end on the new predictions and reproduces a metrics table + confusion matrices + stream-order breakdown.

---

## 10. Risks / watch-outs

- **Statics vs. the paper's thesis** (§2.1) — decide how to frame; consider an ablation with/without statics.
- **`month`/`day` magnitude** dominates the gates when fed raw; after adding drivers/statics you may need to normalize or cyclically encode them so weather isn't drowned out.
- **Adding drivers may barely move wet/dry accuracy** — the target is highly persistent and the current model already ~0.99 via lags/topology. That "no big gain" is itself a legitimate, reportable result; the point is scientific correctness (the "uses meteorology" claim becomes true).
- **Discharge (regression) vs. wet/dry (classification) have different date coverage** (§3). The temporal split is defined on labeled windows; make sure the discharge target still has adequate train/val coverage under the chosen cutoff.
- **Don't reintroduce leakage** via the lagged-obs inputs when defining the split — a val window's 28-day history can reach back into the train period; this is inherent to the temporal split and acceptable, but be explicit about it in the writeup.

---

## 11. Quick reference — model & pipeline internals

- **Feature order (current, 20):** `[etalfalfa, etgrass, prcp, rhmax, rhmin, sph, srad, tmax, tmin, vp, ws, Discharge_CMS_lag_1, Discharge_CMS_lag_7, HoboWetDry0.05_lag_1, HoboWetDry0.05_lag_7, MaxDepth_Censor, MaxDepth_Threshold, MaxDepth_cm, month, day]`. Append statics after this.
- **Targets:** `['HoboWetDry0.05', 'Discharge_CMS']` (classification + log-discharge regression). Note `Flow_Status` is a *third* observation type that was **never a training target** (relevant to a separate "HJFlp" validation idea, out of scope here).
- **Model:** `RGCN_v2` — LSTM gates + `c_t = f_t*(c_t + A @ q_t) + i_t*g_t`; `A` is a registered buffer (793×793, binary, row-normalized, downstream: edge u→v sets `A[v,u]=1`); two linear heads (`cls_head`+sigmoid, `reg_head`). Day-3 prediction = output at the last of 31 timesteps.
- **Hyperparameters (keep unless ablating):** `hidden_dim=64`, `dropout=0.1`, `recur_dropout=0.0`, Adam `lr=1e-3`, `weight_decay=1e-4`, `grad_clip=1.0`, `batch_size=128`, `seq_length=28`, horizon 3, `lambda_discharge=1.0`, `lambda_wetdry=0.5`.
- **Node ordering:** `sorted(graph.nodes())` — must be consistent between `A` and the feature tensor.
```
```

> Handoff prepared on the local machine. To use it on the cluster: commit this file to the new branch (or copy it over) so the next session has it.
