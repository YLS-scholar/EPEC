# EPEC: Efficiency–Prediction Error Coefficient

Reference implementation of the **Efficiency–Prediction Error Coefficient (EPEC)** framework for evaluating time series forecasting models under accuracy–efficiency trade-offs.

EPEC integrates normalized prediction error and training time into a single interpretable score, enabling consistent model ranking and pre‑screening for forecasting tasks.

---

## Repository Structure

```
EPEC/
├── README.md
├── LICENSE
├── requirements.txt
├── EPEC_reference.py        # Minimal reference implementation of the EPEC score
├── EPEC_full.py             # Complete multi‑seed statistical analysis pipeline
├── Beijing_metro.csv    # Publicly released Beijing metro dataset
├── example/
│   ├── all_experiment_results.csv   # Small example input (optional)
│   └── run_example.sh               # One‑click example script
└── output/                  # Generated statistical results (created automatically)
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Use EPEC

### 1. `epec_reference.py` – Minimal Core Implementation

This script provides the core EPEC calculation in a few dozen lines. It is intended for users who already have prediction errors and average training times and want to compute EPEC scores and rankings directly.

**Function signature:**

```python
from epec_reference import epec

epec_scores, rankings = epec(errors, avg_train_times, lamda=0.25)
```

**Parameters:**

- `errors` (`np.ndarray`, shape `(n_models, n_tasks)`): Prediction error (e.g., MSE, MAE, MAPE) for each model and task.
- `avg_train_times` (`np.ndarray`, shape `(n_models,)`): Average training time (seconds) for each model across all tasks.
- `lamda` (`float`, default `0.25`): Exponential decay parameter. Higher values penalise accuracy loss more strongly; lower values favour efficiency. Recommended range: `0.1 – 1.0`.

**Returns:**

- `epec_scores` (`np.ndarray`, shape `(n_models,)`): EPEC score for each model (higher is better).
- `rankings` (`np.ndarray`, shape `(n_models,)`): Rank of each model (1 = highest EPEC).

**Example:**

```python
import numpy as np
from epec_reference import epec

errors = np.array([
    [0.045, 0.139, 0.182],
    [0.052, 0.162, 0.359],
    [0.041, 0.137, 0.364],
    [0.609, 0.225, 0.319],
    [0.054, 0.038, 0.036]
])
avg_train_times = np.array([111.33, 3170.00, 102.52, 7.27, 1843.06])

scores, ranks = epec(errors, avg_train_times, lamda=0.25)
print("EPEC scores:", scores)
print("Rankings (1=best):", ranks)
```

---

### 2. `EPEC_full.py` – Complete Multi‑Seed Analysis Pipeline

This script performs the full statistical analysis used in the paper. It reads raw experiment results across multiple random seeds and multiple datasets, then computes:

- Mean and standard deviation of prediction errors per dataset, model, and horizon.
- Mean and standard deviation of training times per dataset, model, and horizon.
- EPEC values and rankings (mean and standard deviation across seeds).
- Per‑horizon EPEC values.
- Seed‑level rank stability statistics.

**Input file:**

`all_experiment_results.csv` must be placed in the root directory. It must contain the following columns:

| Column      | Description                                      |
|-------------|--------------------------------------------------|
| `seed`      | Random seed (integer)                            |
| `data`      | Dataset name or line name (e.g., `Line1`, `electricity`) |
| `model`     | Model name (e.g., `XGBoost`, `PatchTST`)         |
| `pred_len`  | Forecast horizon (integer)                       |
| `train_time`| Training time in seconds (float)                 |
| `mse`       | Mean squared error (float)                       |
| `mape`      | Mean absolute percentage error (float, optional) |

**Configuration:**

At the top of `EPEC_full.py`, you can adjust:

- `LAMBDA`: EPEC decay parameter (default `0.25`).
- `SEEDS`: List of random seeds (default `[3047, 1234, 5678, 9012, 3456]`).
- `DATASETS`: Dataset definitions (error column, horizons, multi‑line flag).
- `ALL_MODELS`: Expected model names and order.

**Running:**

```bash
python EPEC_full.py
```

**Output files** (saved to `./output/`):

| File | Description |
|------|-------------|
| `seed_error_stats.csv` | Error mean and std per dataset, model, horizon |
| `seed_traintime_stats.csv` | Average and total training time mean and std per dataset, model |
| `seed_traintime_horizons.csv` | Training time per horizon |
| `seed_epec_stats.csv` | EPEC and rank mean and std per dataset, model |
| `seed_epec_per_horizon.csv` | EPEC per horizon |

**Example usage:**

1. Place your `all_experiment_results.csv` in the root directory.
2. Run `python EPEC_full.py`.
3. Inspect the CSV files in `./output/`.

---

## Data

- `data/Beijing_metro.csv` is included in this repository.
- The other four datasets (electricity, exchange_rate, weather, illness) are publicly available. See `data/README.md` for download links.

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{epec2026,
  title   = {Beyond side-by-side comparison: A unified evaluation framework for model selection in time series forecasting},
  author  = {Anonymous Authors},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year    = {2026}
}
```

*(The final citation will be updated upon publication.)*

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
