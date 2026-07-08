# RGCN retrain — evaluation report

Checkpoint: `data/retrain/best_model_retrain_q65.pt` | split: `window_split_map_temporal65.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 332 | 0.973 | 0.989 | 0.982 |
| Day 1 | Headwaters (<=2) | 275 | 0.967 | 0.987 | 0.978 |
| Day 1 | Tailwaters (>=3) | 57 | 1.000 | 1.000 | 1.000 |
| Day 2 | All | 322 | 0.969 | 0.991 | 0.980 |
| Day 2 | Headwaters (<=2) | 265 | 0.966 | 0.989 | 0.977 |
| Day 2 | Tailwaters (>=3) | 57 | 0.982 | 1.000 | 0.989 |
| Day 3 | All | 314 | 0.968 | 0.993 | 0.979 |
| Day 3 | Headwaters (<=2) | 259 | 0.965 | 0.991 | 0.976 |
| Day 3 | Tailwaters (>=3) | 55 | 0.982 | 1.000 | 0.989 |
| All Horizons | All | 968 | 0.970 | 0.991 | 0.980 |
| All Horizons | Headwaters (<=2) | 799 | 0.966 | 0.989 | 0.977 |
| All Horizons | Tailwaters (>=3) | 169 | 0.988 | 1.000 | 0.993 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 486 | 0.959 | 0.990 | 0.971 |
| 2 | 313 | 0.978 | 0.995 | 0.986 |
| 3 | 169 | 0.988 | 1.000 | 0.993 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 223 | 9 |
| **Wet** | 20 | 716 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 349 | 0.959 | 0.896 | 0.3419 | 1675.0 |
| Day 2 | 348 | 0.932 | 0.859 | 0.3067 | 1220.5 |
| Day 3 | 350 | 0.895 | 0.776 | 0.6189 | 1394.5 |
