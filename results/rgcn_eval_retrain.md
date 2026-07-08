# RGCN retrain — evaluation report

Checkpoint: `data/retrain/best_model_retrain.pt` | split: `window_split_map_temporal80.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 200 | 0.950 | 0.981 | 0.967 |
| Day 1 | Headwaters (<=2) | 167 | 0.940 | 0.978 | 0.959 |
| Day 1 | Tailwaters (>=3) | 33 | 1.000 | 1.000 | 1.000 |
| Day 2 | All | 192 | 0.948 | 0.987 | 0.966 |
| Day 2 | Headwaters (<=2) | 159 | 0.943 | 0.986 | 0.962 |
| Day 2 | Tailwaters (>=3) | 33 | 0.970 | 1.000 | 0.982 |
| Day 3 | All | 184 | 0.951 | 0.990 | 0.967 |
| Day 3 | Headwaters (<=2) | 153 | 0.948 | 0.988 | 0.963 |
| Day 3 | Tailwaters (>=3) | 31 | 0.968 | 1.000 | 0.981 |
| All Horizons | All | 576 | 0.950 | 0.986 | 0.966 |
| All Horizons | Headwaters (<=2) | 479 | 0.944 | 0.984 | 0.961 |
| All Horizons | Tailwaters (>=3) | 97 | 0.979 | 1.000 | 0.988 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 295 | 0.939 | 0.981 | 0.957 |
| 2 | 184 | 0.951 | 0.993 | 0.968 |
| 3 | 97 | 0.979 | 1.000 | 0.988 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 129 | 9 |
| **Wet** | 20 | 418 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 295 | 0.957 | 0.847 | 0.3798 | 2021.8 |
| Day 2 | 294 | 0.928 | 0.808 | 0.3391 | 1558.6 |
| Day 3 | 296 | 0.874 | 0.728 | 0.7365 | 1764.4 |
