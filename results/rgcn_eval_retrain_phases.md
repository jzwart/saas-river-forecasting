# RGCN retrain — evaluation report

Checkpoint: `data/retrain/best_model_retrain_phases.pt` | split: `window_split_map_phases.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.951 | 0.977 | 0.970 |
| Day 1 | Headwaters (<=2) | 216 | 0.944 | 0.976 | 0.966 |
| Day 1 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| Day 2 | All | 264 | 0.951 | 0.982 | 0.971 |
| Day 2 | Headwaters (<=2) | 216 | 0.944 | 0.981 | 0.967 |
| Day 2 | Tailwaters (>=3) | 48 | 0.979 | 1.000 | 0.989 |
| Day 3 | All | 257 | 0.957 | 0.986 | 0.974 |
| Day 3 | Headwaters (<=2) | 210 | 0.952 | 0.985 | 0.971 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| All Horizons | All | 784 | 0.953 | 0.982 | 0.972 |
| All Horizons | Headwaters (<=2) | 642 | 0.947 | 0.980 | 0.968 |
| All Horizons | Tailwaters (>=3) | 142 | 0.979 | 1.000 | 0.988 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.930 | 0.983 | 0.957 |
| 2 | 257 | 0.973 | 0.987 | 0.984 |
| 3 | 142 | 0.979 | 1.000 | 0.988 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 109 | 20 |
| **Wet** | 17 | 638 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.983 | 0.969 | 0.0300 | 192.2 |
| Day 2 | 105 | 0.867 | 0.717 | 0.0778 | 201.9 |
| Day 3 | 109 | 0.974 | 0.974 | 0.0340 | 1127.3 |
