# LSTM (All Sites — Mixed Data) — 3-Day Ahead Results

This document summarizes the key results for the mixed-data LSTM, which combines HOBO sensor readings with discretized continuous discharge measurements. This experiment tests whether incorporating more data from continuous gauges improves prediction — the finding is that it causes **catastrophic failure on HOBO sites** due to distributional mismatch.

---

## 1. Training Configuration

- **Dataset**: HOBO sensor wet/dry + discretized continuous discharge (threshold: 0.00014 CMS)
- **Sequences**: 30-day sliding windows, created per site
- **Architecture**: LSTM, hidden_size=80, num_layers=2, dropout=0.188 (Optuna-tuned)
- **Training**: lr=0.000281, batch_size=16, BCEWithLogitsLoss, early stopping patience=3
- **Split**: Random 80/20
- **ADASYN**: Applied to training sequences

---

## 2. HOBO vs Discretized Site Breakdown

| Validation Set | N | Accuracy | ROC-AUC | F1 |
|----------------|---|----------|---------|-----|
| All (HOBO + Discretized) | 31,436 | 0.9279 | 0.9545 | 0.9523 |
| HOBO Only | 2,342 | **0.0491** | 0.5000 | **0.0000** |
| Discretized Only | 29,094 | 0.9986 | 0.9991 | 0.9991 |

### Detailed Classification Report — All

|  | Precision | Recall | F1 | Support |
|--|-----------|--------|-----|---------|
| Dry | 0.742 | 1.000 | 0.852 | 6,519 |
| Wet | 1.000 | 0.909 | 0.952 | 24,917 |

### Detailed Classification Report — HOBO Only

|  | Precision | Recall | F1 | Support |
|--|-----------|--------|-----|---------|
| Dry | 0.049 | 1.000 | 0.094 | 115 |
| Wet | 0.000 | 0.000 | 0.000 | 2,227 |

### Detailed Classification Report — Discretized Only

|  | Precision | Recall | F1 | Support |
|--|-----------|--------|-----|---------|
| Dry | 0.994 | 1.000 | 0.997 | 6,404 |
| Wet | 1.000 | 0.998 | 0.999 | 22,690 |

The model achieves near-perfect performance on discretized continuous observations (0.999 accuracy) but **completely fails on HOBO sensor sites (0.049 accuracy, F1 0.000)** — worse than random guessing — despite HOBO sites being the primary target for headwater prediction. The overall accuracy of 0.928 masks this catastrophic failure because HOBO sites represent only 7.4% of the validation set.

The model predicts every HOBO observation as "dry" (recall 0.000 for wet class), indicating it has learned a decision boundary that separates the two data distributions rather than learning actual wet/dry hydrology.

---

## 3. Performance by Stream Order

### Headwaters vs Tailwaters

| Group | N | Accuracy | ROC-AUC | F1 |
|-------|---|----------|---------|-----|
| Headwaters (order ≤ 2) | 12,380 | 0.8174 | 0.9074 | 0.8980 |
| Tailwaters (order ≥ 3) | 12,814 | 0.9995 | 0.9997 | 0.9997 |

### Per Stream Order

| Stream Order | N | Accuracy | ROC-AUC | F1 |
|-------------|---|----------|---------|-----|
| 1 | 3,374 | 0.9935 | 0.9967 | 0.9967 |
| 2 | 9,006 | **0.7514** | 0.8738 | 0.8556 |
| 3 | 6,470 | 0.9991 | 0.9991 | 0.9991 |
| 4 | 3,108 | 1.0000 | NaN | 1.0000 |
| 5 | 25 | 1.0000 | NaN | 1.0000 |
| 6 | 3,211 | 1.0000 | NaN | 1.0000 |

### Detailed Classification Report — Order 1

|  | Precision | Recall | F1 | Support |
|--|-----------|--------|-----|---------|
| Dry | 0.593 | 1.000 | 0.744 | 32 |
| Wet | 1.000 | 0.993 | 0.997 | 3,342 |

### Detailed Classification Report — Order 2

|  | Precision | Recall | F1 | Support |
|--|-----------|--------|-----|---------|
| Dry | 0.056 | 1.000 | 0.107 | 134 |
| Wet | 1.000 | 0.748 | 0.856 | 8,872 |

Order 2 headwater streams show the most severe degradation (0.751 accuracy), aligned with where HOBO sensors are concentrated in the network. The model predicts nearly all order 2 wet observations as dry because it has learned the discretized discharge distribution — which is fundamentally different from the HOBO sensor distribution at these sites.

---

## 4. Key Takeaway: Distributional Mismatch

Field-classified wet/dry from HOBO sensors has fundamentally different distributional characteristics than threshold-based discretization of gauge measurements. Naive data combination is harmful:

- The mixed-data LSTM learns to separate the two data distributions rather than the wet/dry boundary.
- HOBO site accuracy collapses from 0.977 (HOBO-only LSTM) to 0.049 (mixed-data LSTM).
- Overall metrics (0.928 accuracy) mask the catastrophic failure because discretized sites dominate the dataset (93%).

The RGCN avoids this problem entirely through its multi-output architecture — jointly predicting both observation types with separate output heads while sharing spatial and temporal representations. This architectural solution to data heterogeneity outperforms simple data mixing.
