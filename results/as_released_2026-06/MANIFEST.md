# As-Released RGCN Baseline — Snapshot 2026-06

This directory preserves the **as-released** RGCN model, split, graph, and eval so
the paper's current numbers stay reproducible while the `rgcn-retrain` branch
regenerates everything under new filenames. See `RGCN_RETRAIN_PLAN.md` §8.

**Do not overwrite or delete anything referenced here.** All retrain outputs use
new filenames (e.g. `best_model_retrain.pt`, `hja_graph_drivers.gpickle`,
`window_split_map_temporal80.csv`).

## Provenance

- **Released code commit:** `51b140e` — tagged **`results-as-released`**.
- **Derived artifacts:** Hugging Face `michaeltm365/saas-river-forecasting` (main).
- **Raw public data:** ScienceBase DOI `10.5066/P19R5TXW`, item `6977e36dd4be02609dd04095`.

## Released artifacts (physical local copies, MD5)

| File | Size (bytes) | MD5 |
|---|---|---|
| `best_model.pt` | 2840866 | `4df44baf37127cdd18fbbb2b1d4f5eb5` |
| `hja_graph.gpickle` | 15378100 | `d7184fd87406e96f2525548b4ba92420` |
| `window_split_map.csv` | 49144 | `b202b79d3066bf8c6eaacaf732d3b982` |
| `rgcn_eval.ipynb` (with outputs) | 2493815 | copy of `rgcn/rgcn_eval.ipynb` @ `51b140e` |

The binary snapshots above are **git-ignored** (the repo keeps data on HF/ScienceBase,
not in git). This manifest + the `results-as-released` tag + the HF/ScienceBase
sources are the durable record; the local copies are anti-clobber insurance.

## Released predictions — NOT copied locally

`train_val_predictions_day{1,2,3}.csv` (1.99 GB each) were not downloaded. Fetch
from HF when needed to reproduce released metrics:

```bash
python download_data.py --include predictions
```

## Not present

- `data/result_summaries/rgcn_eval_results.md` (referenced in plan §8) does not
  exist in this checkout; released metrics are reproducible by running
  `rgcn/rgcn_eval.ipynb` against the released predictions above.
