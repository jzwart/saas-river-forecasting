# RGCN retrain — evaluation report

Checkpoint: `data/retrain/best_model_retrain.pt` | split: `window_split_map_temporal80.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 200 | 0.945 | 0.983 | 0.964 |
| Day 1 | Headwaters (<=2) | 167 | 0.934 | 0.980 | 0.955 |
| Day 1 | Tailwaters (>=3) | 33 | 1.000 | 1.000 | 1.000 |
| Day 2 | All | 192 | 0.948 | 0.989 | 0.966 |
| Day 2 | Headwaters (<=2) | 159 | 0.943 | 0.987 | 0.962 |
| Day 2 | Tailwaters (>=3) | 33 | 0.970 | 1.000 | 0.982 |
| Day 3 | All | 184 | 0.951 | 0.991 | 0.967 |
| Day 3 | Headwaters (<=2) | 153 | 0.948 | 0.990 | 0.964 |
| Day 3 | Tailwaters (>=3) | 31 | 0.968 | 1.000 | 0.981 |
| All Horizons | All | 576 | 0.948 | 0.988 | 0.966 |
| All Horizons | Headwaters (<=2) | 479 | 0.942 | 0.986 | 0.960 |
| All Horizons | Tailwaters (>=3) | 97 | 0.979 | 1.000 | 0.988 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 295 | 0.936 | 0.981 | 0.954 |
| 2 | 184 | 0.951 | 0.995 | 0.969 |
| 3 | 97 | 0.979 | 1.000 | 0.988 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 125 | 13 |
| **Wet** | 17 | 421 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 295 | 0.958 | 0.857 | 0.3731 | 1887.0 |
| Day 2 | 294 | 0.926 | 0.797 | 0.3454 | 1395.4 |
| Day 3 | 296 | 0.891 | 0.754 | 0.6860 | 1557.1 |
