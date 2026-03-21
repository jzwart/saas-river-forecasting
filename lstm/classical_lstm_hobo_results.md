# Classical Models & LSTM (HOBO-Only) — 3-Day Ahead Results

This document summarizes the key results for Logistic Regression, XGBoost, and the discrete-only LSTM, all trained on **HOBO sensor data only** with a **3-day ahead prediction horizon** (predict wet/dry at t+3 given features at t).

All three models use ADASYN oversampling to address class imbalance (~82% wet, ~18% dry).

---

## 1. Logistic Regression

### Overall Performance by Split Strategy

| Split | Accuracy | ROC-AUC | F1 |
|-------|----------|---------|-----|
| Random | 0.9499 | 0.9613 | 0.9683 |
| Temporal | 0.9324 | 0.9044 | 0.9573 |
| Site-Based | 0.7958 | 0.7163 | 0.8622 |

**Spatial transfer gap**: 0.154 (Random → Site-Based accuracy drop).

### Detailed Classification Report — Temporal Split

|  | Precision | Recall | F1 | Support |
|--|-----------|--------|-----|---------|
| Dry | 0.820 | 0.857 | 0.838 | 154 |
| Wet | 0.963 | 0.952 | 0.957 | 600 |

### Stream Order Breakdown

#### Random Split

| Order | N | Accuracy | ROC-AUC | F1 |
|-------|---|----------|---------|-----|
| 1 | 271 | 0.9299 | 0.9493 | 0.9549 |
| 2 | 145 | 0.9862 | 0.9916 | 0.9915 |
| 3 | 103 | 0.9515 | 0.9509 | 0.9693 |

#### Temporal Split

| Order | N | Accuracy | ROC-AUC | F1 |
|-------|---|----------|---------|-----|
| 1 | 387 | 0.9380 | 0.9089 | 0.9582 |
| 2 | 230 | 0.9435 | 0.9217 | 0.9673 |
| 3 | 137 | 0.8978 | 0.8840 | 0.9369 |

#### Site-Based Split

| Order | N | Accuracy | ROC-AUC | F1 |
|-------|---|----------|---------|-----|
| 1 | 340 | 0.9618 | 0.9462 | 0.9733 |
| 2 | 115 | **0.0957** | 0.5000 | **0.1746** |
| 3 | 118 | 1.0000 | NaN | 1.0000 |

The site-based split reveals **catastrophic failure on order 2 streams** (accuracy 0.096, F1 0.175), while order 1 accuracy remains high (0.962). This asymmetry confirms the model relies on site-specific features that transfer very differently across stream orders.

### Feature Importance (Coefficient Magnitude)

| Rank | Feature | Coefficient |
|------|---------|-------------|
| 1 | out_degree | 4.087 |
| 2 | lagged_target | 3.294 |
| 3 | Slope | -3.153 |
| 4 | aspect_se_pct | 2.853 |
| 5 | aspect_ne_pct | -2.657 |
| 6 | sph | 2.485 |
| 7 | rhmax | -1.952 |
| 8 | rhmin | 1.924 |
| 9 | elev_min_cm | -1.595 |
| 10 | srad | 1.563 |

---

## 2. XGBoost

### Overall Performance by Split Strategy

| Split | Accuracy | ROC-AUC | F1 |
|-------|----------|---------|-----|
| Random | 0.9788 | 0.9674 | 0.9869 |
| Temporal | 0.9615 | 0.9517 | 0.9757 |
| Site-Based | 0.8639 | 0.8824 | 0.8860 |

**Spatial transfer gap**: 0.115 (Random → Site-Based accuracy drop).

### Detailed Classification Report — Temporal Split

|  | Precision | Recall | F1 | Support |
|--|-----------|--------|-----|---------|
| Dry | 0.883 | 0.935 | 0.909 | 154 |
| Wet | 0.983 | 0.968 | 0.976 | 600 |

### Detailed Classification Report — Site-Based Split

|  | Precision | Recall | F1 | Support |
|--|-----------|--------|-----|---------|
| Dry | 0.741 | 0.946 | 0.831 | 203 |
| Wet | 0.965 | 0.819 | 0.886 | 370 |

### Stream Order Breakdown

#### Random Split

| Order | N | Accuracy | ROC-AUC | F1 |
|-------|---|----------|---------|-----|
| 1 | 271 | 0.9668 | 0.9501 | 0.9794 |
| 2 | 145 | 0.9931 | 0.9958 | 0.9958 |
| 3 | 103 | 0.9903 | 0.9750 | 0.9940 |

#### Temporal Split

| Order | N | Accuracy | ROC-AUC | F1 |
|-------|---|----------|---------|-----|
| 1 | 387 | 0.9509 | 0.9391 | 0.9664 |
| 2 | 230 | 0.9696 | 0.9673 | 0.9825 |
| 3 | 137 | 0.9781 | 0.9870 | 0.9868 |

#### Site-Based Split

| Order | N | Accuracy | ROC-AUC | F1 |
|-------|---|----------|---------|-----|
| 1 | 340 | 0.7971 | 0.8479 | 0.8353 |
| 2 | 115 | 0.9217 | 0.9161 | 0.6897 |
| 3 | 118 | 1.0000 | NaN | 1.0000 |

XGBoost shows more consistent degradation under site-based evaluation than LR. Order 1 headwater streams — the most abundant and ecologically critical — drop to 0.797 accuracy.

### Feature Importance (Gain)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | lagged_target | 0.7831 |
| 2 | elev_mean_cm | 0.0487 |
| 3 | out_degree | 0.0328 |
| 4 | etalfalfa | 0.0215 |
| 5 | elev_max_cm | 0.0182 |
| 6 | aspect_ne_pct | 0.0093 |
| 7 | aspect_se_pct | 0.0080 |
| 8 | etgrass | 0.0077 |
| 9 | curv_mean | 0.0074 |
| 10 | rhmin | 0.0069 |

Lagged target dominates with >0.78 relative importance; all other features contribute minimally. The model primarily learns persistence conditioned on site identity.

---

## 3. LSTM (HOBO-Only, Random Split)

Trained on HOBO sensor observations only (same data as classical models). Uses a 30-day sliding window of features. Sequences created per site to avoid cross-site boundary contamination. ADASYN applied to training sequences.

- **Dataset**: 2,593 samples → 1,933 sequences (30-day windows)
- **Class distribution**: 1,585 wet / 348 dry sequences (81.7% wet)
- **After ADASYN**: 2,528 training sequences
- **Architecture**: LSTM, hidden_size=64, num_layers=2, dropout=0.3
- **Training**: 15 epochs, lr=1e-4, BCEWithLogitsLoss, early stopping patience=5

### Overall Performance (Random Split)

| Accuracy | ROC-AUC | F1 |
|----------|---------|-----|
| 0.9767 | 0.9748 | 0.9856 |

### Detailed Classification Report

|  | Precision | Recall | F1 | Support |
|--|-----------|--------|-----|---------|
| Dry | 0.908 | 0.972 | 0.939 | 71 |
| Wet | 0.994 | 0.978 | 0.986 | 316 |

ADASYN improved dry recall from ~0.60 (baseline without resampling) to 0.972 while maintaining high wet precision (0.994).

### Feature Importance (Permutation, F1 Drop)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | lagged_target | 0.0508 |
| 2 | aspect_ne_pct | 0.0302 |
| 3 | out_degree | 0.0297 |
| 4 | aspect_se_pct | 0.0157 |
| 5 | sph | 0.0145 |
| 6 | slp_median_pct | 0.0097 |
| 7 | elev_min_cm | 0.0096 |
| 8 | aspect_nw_pct | 0.0081 |
| 9 | srad | 0.0066 |
| 10 | slp_mean_pct | 0.0064 |

The LSTM shows broader feature utilization than XGBoost: lagged target is still most important but does not dominate as severely. Static features (aspect, out_degree) show moderate importance, and dynamic features (sph, srad) contribute detectably — indicating the LSTM captures temporal patterns integrating both persistence and meteorological drivers.

---

## 4. Cross-Model Comparison (Temporal Split)

| Model | Accuracy | ROC-AUC | F1 | Dry Precision | Dry Recall | Dry F1 |
|-------|----------|---------|-----|---------------|------------|--------|
| Logistic Regression | 0.9324 | 0.9044 | 0.9573 | 0.820 | 0.857 | 0.838 |
| XGBoost | 0.9615 | 0.9517 | 0.9757 | 0.883 | 0.935 | 0.909 |
| LSTM (HOBO only)* | 0.9767 | 0.9748 | 0.9856 | 0.908 | 0.972 | 0.939 |

*LSTM uses random split (temporal split not evaluated for LSTM); included for reference.

Key observations:
- All models achieve >0.93 accuracy on standard evaluation, but this masks the static feature memorization problem revealed by site-based splits.
- The LSTM achieves the best dry class detection (F1 0.939 vs XGBoost 0.909 vs LR 0.838), largely driven by ADASYN-enhanced dry recall.
- Under site-based evaluation (LR and XGBoost only), performance degrades dramatically — confirming that classical models memorize static site characteristics rather than learning transferable hydrological dynamics.
