# Evaluating Data-Driven Models for Multimodal Prediction of Headwater Streamflow

[![USGS](https://img.shields.io/badge/USGS-Collaboration-green)](https://www.usgs.gov/)

## Overview

This repository contains the code and analysis for our research on machine learning approaches for streamflow forecasting in sparse observational networks. Using the H.J. Andrews Forest Long-Term Ecological Research dataset, we evaluate and compare four supervised learning models for predicting wet/dry streamflow status:

- **Logistic Regression** - Baseline classical approach
- **XGBoost** - Gradient-boosted decision trees
- **LSTM** - Long Short-Term Memory neural networks for temporal modeling
- **RGCN** - Recurrent Graph Convolutional Networks for spatial-temporal modeling

## Repository Structure

```
├── lr/
│   └── lr.ipynb                     # Logistic Regression: training, evaluation, inference
├── xgb/
│   └── xgb.ipynb                    # XGBoost: training, evaluation, inference
├── lstm/
│   ├── lstm_hobo_sites.ipynb        # LSTM on HOBO sensor sites only (discrete observations)
│   ├── lstm_all_sites.ipynb         # LSTM on HOBO + discretized discharge (mixed data)
│   └── lstm_all_sites_results.md    # Mixed-data LSTM results (distributional mismatch finding)
├── rgcn/
│   ├── pipeline/                    # Reproducible RGCN retrain pipeline — see rgcn/pipeline/README.md
│   ├── config.yml                   # Retrain config (+ config_q65.yml / config_phases.yml split variants)
│   ├── inspect_driver_weights.py    # Verify drivers/statics are active in a checkpoint
│   ├── build_graph.ipynb            # [as-released] Stream network graph construction
│   ├── train_gnn.ipynb              # [as-released] RGCN model training
│   ├── rgcn_eval.ipynb              # RGCN evaluation: metrics, stream order, perennial status
│   ├── rgcn_eval_results.md         # [as-released] RGCN evaluation results summary
│   └── rgcn_config.yaml             # [as-released] RGCN model configuration
├── synthetic_data/
│   └── gam.ipynb                    # GAM-based synthetic data generation
├── results/
│   ├── rgcn_eval_retrain*.md        # Retrained-RGCN metrics (per split variant)
│   └── as_released_2026-06/         # Manifest of the archived released baseline (tag: results-as-released)
├── download_data.py                 # Fetch ScienceBase + Hugging Face data into data/
├── classical_lstm_hobo_results.md   # LR, XGBoost, LSTM (HOBO-only) results summary
└── README.md
```

> **Retraining the RGCN:** the released RGCN had three defects (unused
> meteorological drivers, unfed static features, and normalization/split
> leakage). `rgcn/pipeline/` retrains it with the fixes on honest temporal
> splits — full reproduction instructions in
> [`rgcn/pipeline/README.md`](rgcn/pipeline/README.md). The as-released
> baseline is preserved untouched (git tag `results-as-released`,
> `results/as_released_2026-06/MANIFEST.md`).

## Model Weights

Pre-trained model weights and processed data are available on Hugging Face:

🤗 **[michaeltm365/saas-river-forecasting](https://huggingface.co/michaeltm365/saas-river-forecasting)**

### Quick Download
```python
from huggingface_hub import hf_hub_download

# Download RGCN model weights
model_path = hf_hub_download(
    repo_id="michaeltm365/saas-river-forecasting", 
    filename="best_model.pt"
)

# Download graph structure
graph_path = hf_hub_download(
    repo_id="michaeltm365/saas-river-forecasting", 
    filename="hja_graph.gpickle"
)
```

**Files available:**
- `best_model.pt` - Pre-trained RGCN model weights  
- `hja_graph.gpickle` - H.J. Andrews stream network topology
- `hja_edge_index.npz` - Graph connectivity matrix
- `static_vars_pivot.csv` - Watershed characteristics
- Additional supporting data files

## Data

This project uses data from the H.J. Andrews Forest Long-Term Ecological Research site, including:

- **Observational data**: Continuous discharge measurements and discrete wet/dry classifications
- **Driver variables**: Meteorological data from GridMET (precipitation, temperature, humidity, etc.)
- **Static variables**: Watershed characteristics (slope, elevation, aspect, drainage area)
- **Network topology**: NHDPlus stream segment connectivity

**Note**: Data files are not included in this repository. Please contact the authors or USGS for data access.

## Data Augmentations

1. **Discharge Discretization**: Threshold-based conversion (0.00014 CMS) of continuous measurements to binary wet/dry
2. **Time-Series ADASYN**: Adapted resampling preserving temporal autocorrelation within sliding windows
3. **Synthetic Data via GAMs**: Generalized Additive Models for augmenting sparse observation sites (RMSE=1.33)

## Evaluation Framework

Three train-test splitting strategies for Logistic Regression and XGBoost:

| Strategy | Description | Tests |
|----------|-------------|-------|
| Random | Standard ML benchmarking | General performance |
| Temporal | Chronological split | Forecasting ability |
| Site-based | Entire sites withheld | Spatial generalizability |

## Requirements

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and locked in
`uv.lock` (Python version in `.python-version`). Set up the environment with:

```bash
uv sync
```

On Linux, torch is pinned to the CUDA 12.6 build (runs on CUDA 12.4+ drivers);
other platforms get the default PyPI wheels. RGCN training requires a Linux
CUDA GPU. Fetch data with `uv run python download_data.py` (one 2.13 GB file,
`met_drivers.csv`, requires a manual browser download — the script prints the
URL).

## Usage

All notebooks follow a consistent structure and use standardized variable naming:

1. **Imports**
2. **Data Preprocessing** — Loading, merging, and creating the `central_df` dataframe with standardized column names (`wetdry_status` for the current observation, `wet_dry_next` for the prediction target)
3. **Model Training** — With ADASYN class imbalance handling
4. **Evaluation** — Metrics, classification reports, confusion matrices, and feature importance
5. **Inference** — Function for predicting wet/dry status at new site-date combinations

Example inference (LR/XGBoost):
```python
predict_site_date(
    model=model,
    central_df=central_df,
    site_id="HoboSite100",
    date="2020-10-22"
)
# Output: "Site HoboSite100 on 2020-10-25 (predicted from 2020-10-22): DRY, (P(wet)=0.0000)"
```

## Citation

If you use this code or findings in your research, please cite:

```
Huang, A., Prieto, C., Murphy, M., Kandadai, A., Wang, A., Yu, A., Krishnan, A.,
Danes, A., Wong, A., Patel, K., Iyer, S., Dubey, V., Nguyen, V., Zwart, J.,
Cook, G., & Chelgren, N. (2025). Evaluating Data-Driven Models for Multimodal Prediction of Headwater Streamflow.
[Preprint in preparation]
```

## Authors

**Student Association for Applied Statistics (SAAS), UC Berkeley**
- Alex Huang, Cristina Prieto, Michael Murphy, Akshath Kandadai, Allison Wang, Amber Yu, Anika Krishnan, Anya Danes, Audrey Wong, Krish Patel, Sanika Iyer, Viksar Dubey, Vivian Nguyen

**United States Geological Survey (USGS)**
- Jacob Zwart, Gericke Cook, Nathan Chelgren

## Acknowledgements

We thank the U.S. Geological Survey for data access and collaboration, and the H.J. Andrews Forest Long-Term Ecological Research program for maintaining the observational network.

---

*For questions or collaboration inquiries, please open an issue or contact the authors.*
