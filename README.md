# Payment Fraud Detection & Transaction Monitoring

End-to-end machine learning system for classifying credit-card transactions as **legitimate** or **fraudulent**, with data-quality validation, leakage-safe feature engineering, multi-model training, **cost-aware decision thresholds**, **SHAP explainability**, a Streamlit monitoring dashboard, and GitHub Actions CI.

Built on the public **[ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** dataset (`creditcard.csv`, ~285k transactions).

---

## Highlights

| Area | Implementation |
|------|----------------|
| Dataset | Public ULB / Kaggle `creditcard.csv` (≥ 100k rows) |
| Data quality | **12** validation rules + markdown/JSON report |
| Features | **48** leakage-safe features (amount, time, velocity, PCA) |
| Models | Logistic Regression, Random Forest, tuned **XGBoost** |
| Imbalance | `class_weight` / `scale_pos_weight` |
| Evaluation | Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix |
| Decisions | F1-optimal **and** **cost-optimal** thresholds |
| Explainability | Native importance + **SHAP** (dashboard + artifacts) |
| Dashboard | Streamlit (overview, cost, SHAP, predictions) |
| Storage | SQLite + Tableau-compatible CSVs |
| Tests | **18** PyTest cases |
| CI | GitHub Actions (`pytest` on push/PR) |

All metrics below are **measured from an actual pipeline run** — none are fabricated. Source of truth: [`metrics_summary.json`](./metrics_summary.json).

---

## Architecture

```text
creditcard.csv  (Kaggle / ULB)
        │
        ▼
┌───────────────────┐
│  Ingest + standardize │  → data/raw/
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  Data-quality (12)    │  → reports/data_quality_report.md
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  EDA + figures        │  → reports/figures/, data/processed/eda_*.csv
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  Feature engineering  │  → 48 features (shift/rolling, no label leakage)
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  Train / Val / Test   │  temporal 70% / 15% / 15% by Time
│  LR · RF · XGBoost    │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  Cost threshold + SHAP│  → artifacts/, reports/figures/
└─────────┬─────────┘
          ▼
   metrics_summary.json
   Streamlit dashboard
   SQLite (fraud_monitoring.db)
```

---

## Dataset

**Source:** [Credit Card Fraud Detection (ULB / Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place the file at the project root (gitignored — do **not** commit it):

```text
payment-fraud-detection/creditcard.csv
```

| Column | Description |
|--------|-------------|
| `Time` | Seconds elapsed between each transaction and the first |
| `V1`–`V28` | PCA-transformed features (anonymized) |
| `Amount` | Transaction amount |
| `Class` | `1` = fraud, `0` = legitimate |

### Schema gaps (documented, not fabricated)

This public file does **not** include location, device type, payment method, customer ID, or merchant ID. The pipeline does **not** invent those fields. Monitoring uses **hour-of-day** (from `Time`) and **amount bins** instead. Frequency / historical behavior are derived from the time-ordered stream.

See [`reports/data_dictionary.md`](./reports/data_dictionary.md).

---

## Quick start

### Requirements

- Python **3.11**
- macOS + XGBoost: `brew install libomp`

### Setup

```bash
cd payment-fraud-detection
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `creditcard.csv` from Kaggle and place it in the project root.

### Run the full pipeline

```bash
python run_pipeline.py
# optional: python run_pipeline.py --creditcard /path/to/creditcard.csv
# reuse existing data/raw: python run_pipeline.py --skip-ingest
```

### Tests

```bash
pytest tests/ -v
```

### Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard tabs:

1. **Overview** — KPIs, fraud by hour / amount bin, confusion matrix, model table  
2. **Cost-aware threshold** — F1 vs cost policy, $ cost curve  
3. **Explainability (SHAP)** — mean \|SHAP\| and native importance  
4. **Predictions** — high-risk test-set scores  

---

## Project structure

```text
payment-fraud-detection/
├── .github/workflows/ci.yml      # GitHub Actions CI
├── creditcard.csv                # gitignored — download from Kaggle
├── run_pipeline.py               # end-to-end entrypoint
├── metrics_summary.json          # measured metrics from last run
├── requirements.txt
├── README.md
├── dashboard/
│   └── app.py                    # Streamlit monitoring UI
├── notebooks/
│   └── 01_eda.ipynb
├── data/
│   ├── raw/                      # standardized raw (gitignored)
│   ├── processed/                # clean, features, SQLite, Tableau CSVs
│   └── external/
├── src/
│   ├── data/
│   │   ├── load_creditcard.py    # ingest + standardize ULB CSV
│   │   ├── ingest.py             # load / SQLite helpers
│   │   ├── validate.py           # 12 DQ rules + reports
│   │   └── eda.py                # EDA tables + figures
│   ├── features/
│   │   └── engineering.py        # 48 leakage-safe features
│   ├── models/
│   │   ├── train.py              # LR / RF / XGBoost + evaluation
│   │   ├── cost_threshold.py     # $ cost-aware threshold search
│   │   └── explain.py            # SHAP + native importance
│   └── utils/
│       └── paths.py
├── tests/
│   └── test_pipeline.py          # 18 PyTest cases
├── models/                       # .joblib artifacts (gitignored)
├── artifacts/                    # predictions, SHAP, cost curve
└── reports/
    ├── data_dictionary.md
    ├── data_quality_report.md
    ├── feature_dictionary.md
    ├── figures/
    └── metrics/
```

---

## 1. Data ingestion

- Loads `creditcard.csv` via `src/data/load_creditcard.py`
- Standardizes to project schema: `transaction_id`, `seconds_from_start`, `timestamp`, `amount`, `amount_bin`, `hour`, `is_fraud`, `V1`–`V28`
- Persists raw standardized data under `data/raw/` (CSV + Parquet), separate from `data/processed/`
- Optional SQLite warehouse: `data/processed/fraud_monitoring.db`

---

## 2. Data-quality validation (12 rules)

Implemented in `src/data/validate.py`. Violating rows are removed for critical/warning rules; info rules are flagged only.

| ID | Rule | Action |
|----|------|--------|
| DQ01 | Missing critical fields | Remove |
| DQ02 | Missing PCA features V1–V28 | Remove |
| DQ03 | Duplicate transaction IDs | Remove extras |
| DQ04 | Duplicate feature rows | Remove extras |
| DQ05 | Negative / null amounts | Remove |
| DQ06 | Invalid `Time` values | Remove |
| DQ07 | Invalid timestamps | Remove |
| DQ08 | Invalid class labels (must be 0/1) | Remove |
| DQ09 | Non-finite PCA values | Remove |
| DQ10 | Amount / label null inconsistency | Remove |
| DQ11 | Extreme amount outliers (p99.9) | Flag only |
| DQ12 | Extreme PCA outliers (p99.9) | Flag only |

Report: [`reports/data_quality_report.md`](./reports/data_quality_report.md)

**Measured (last run):** 284,807 raw → **283,726** clean (**1,081** duplicates removed).

---

## 3. Exploratory data analysis

`src/data/eda.py` + `notebooks/01_eda.ipynb`:

- Class balance and fraud rate  
- Amount distribution by class  
- Fraud rate by **hour** and **amount bin**  
- Summary tables exported as Tableau-friendly CSVs under `data/processed/`  
- Figures under `reports/figures/`  

---

## 4. Feature engineering (48 features)

`src/features/engineering.py` — leakage controls use `shift` / expanding / rolling so the current row (and label) never enter its own aggregates. `is_fraud` / `Class` are **never** model inputs.

| Group | Examples |
|-------|----------|
| Transaction | `amount`, `log_amount`, `amount_sqrt` |
| Temporal | `hour`, `hour_sin/cos`, `day_index`, `seconds_from_start` |
| Velocity / behavior | `seconds_since_prev`, `amount_delta_prev`, expanding mean/std/z-score, rolling mean/std/rate |
| PCA aggregates | `pca_l2_norm`, `pca_abs_max/mean/std` |
| PCA components | `V1`–`V28` |

Full definitions: [`reports/feature_dictionary.md`](./reports/feature_dictionary.md)

---

## 5. Model development

`src/models/train.py`

| Step | Detail |
|------|--------|
| Split | Temporal **70 / 15 / 15** (train / val / test) by `Time` |
| Models | Logistic Regression (+ `StandardScaler`), Random Forest, XGBoost |
| Imbalance | `class_weight="balanced"` / `balanced_subsample` / `scale_pos_weight` |
| Tuning | XGBoost small grid search; select by validation **PR-AUC** |
| Ranking | Best model by **test PR-AUC** |
| Thresholds | Val max-**F1** threshold + **cost-minimizing** threshold (see below) |
| Metrics | Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix |

Artifacts: `models/*.joblib`, `artifacts/test_predictions.csv`

---

## 6. Cost-aware thresholding

`src/models/cost_threshold.py`

| Event | Default cost |
|-------|----------------|
| False positive | **$10** (review / customer friction) |
| False negative | **missed transaction amount** (min $1) |

On validation, the pipeline sweeps thresholds and picks the one with **lowest total $ cost**, then freezes it for test. The dashboard compares F1-optimal vs cost-optimal policies.

### Measured cost comparison (test set, last run)

| Policy | Threshold | FP | FN | Total $ cost |
|--------|-----------|----|----|--------------|
| F1-optimal | 0.95 | 56 | 12 | **$2,921.09** |
| Cost-optimal | 0.99 | 26 | 12 | **$2,621.09** |
| **Savings** | | | | **$300.00** |

Operational dashboard predictions use the **cost-optimal** threshold.

---

## 7. Explainability (SHAP)

`src/models/explain.py`

- Native importance (`coef_` for LR, `feature_importances_` for trees)  
- Mean \|SHAP\| on up to **800** test samples  
  - `LinearExplainer` for Logistic Regression  
  - `TreeExplainer` for Random Forest / XGBoost  
- CSV + PNG artifacts; Streamlit **Explainability** tab  

**Top SHAP features (last run, Logistic Regression):** `amount_sqrt`, `log_amount`, `V14`, `V2`, `V4`, …

---

## 8. Monitoring dashboard

```bash
streamlit run dashboard/app.py
```

Shows:

- Total transactions, fraud rate  
- Precision / recall / F1 / ROC-AUC / PR-AUC  
- False positives  
- Fraud by hour and amount bin  
- Confusion matrix  
- Cost curve and F1 vs cost policy table  
- SHAP / feature importance charts  
- High-risk prediction table  

---

## 9. Testing & CI

### PyTest (`tests/test_pipeline.py` — 18 tests)

- Data validation rules (≥ 10 rules, amounts, duplicates, labels, PCA)  
- Feature engineering (≥ 20 features, no label leakage, causal expanding stats)  
- Temporal split ordering + model matrix shapes  
- Prediction output contract  
- Cost-threshold search + F1 vs cost comparison  
- Native feature importance helper  

```bash
pytest tests/ -v
```

### GitHub Actions

[`.github/workflows/ci.yml`](./.github/workflows/ci.yml) runs on push/PR to `main`:

1. Checkout  
2. Setup Python 3.11  
3. `pip install -r requirements.txt`  
4. `pytest tests/ -v`  

---

## Measured model results (last run)

**Dataset:** 284,807 raw → 283,726 clean · fraud rate **0.167%** · **48** features  
**Split:** train 198,608 · val 42,559 · test 42,559  

### Test metrics at F1-optimal thresholds

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP |
|-------|-----------|--------|-----|---------|--------|-----|
| **Logistic Regression** (best PR-AUC) | **0.417** | **0.769** | **0.541** | **0.978** | **0.770** | **56** |
| Random Forest | 0.881 | 0.712 | 0.787 | 0.964 | 0.759 | 5 |
| XGBoost (tuned) | 0.804 | 0.712 | 0.755 | 0.977 | 0.767 | 9 |

### Cost-optimal policy (Logistic Regression, thr=0.99)

| Metric | Value |
|--------|-------|
| Precision | 0.606 |
| Recall | 0.769 |
| F1 | 0.678 |
| ROC-AUC | 0.978 |
| PR-AUC | 0.770 |
| Confusion | TN 42481 · FP 26 · FN 12 · TP 40 |

Re-run the pipeline to refresh numbers; always trust `metrics_summary.json` over older README tables if they diverge.

---

## Tech stack

- **Python 3.11** · Pandas · NumPy · scikit-learn · XGBoost · SHAP  
- **SQLite** · Streamlit · Plotly · Matplotlib · Seaborn  
- **PyTest** · Jupyter · GitHub Actions  

---

## Limitations

- ULB data has no customer / merchant / geo / device identifiers  
- Behavioral features are **stream-level**, not per-customer  
- Two-day European card snapshot — not a live production feed  
- Default $ costs are illustrative; tune for your business  
- PCA features are not human-interpretable (SHAP still ranks them)  

---

## Future improvements

- Entity-level / graph features on richer payment datasets  
- Drift detection (PSI) and scheduled retrain  
- FastAPI `/predict` scoring service  
- Docker Compose one-command demo  
- Calibration plots and PR curves as committed report assets  

---

## License

Educational / portfolio use. Dataset license follows the [Kaggle ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) terms.
