"""Project path helpers and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# Public ULB Credit Card Fraud dataset (user-provided)
CREDITCARD_CSV = PROJECT_ROOT / "creditcard.csv"
CREDITCARD_EXTERNAL = DATA_EXTERNAL / "creditcard.csv"

RAW_CSV = DATA_RAW / "transactions_raw.csv"
RAW_PARQUET = DATA_RAW / "transactions_raw.parquet"
CLEAN_CSV = DATA_PROCESSED / "transactions_clean.csv"
FEATURES_CSV = DATA_PROCESSED / "transactions_features.csv"
FEATURES_PARQUET = DATA_PROCESSED / "transactions_features.parquet"
DB_PATH = DATA_PROCESSED / "fraud_monitoring.db"
METRICS_SUMMARY = PROJECT_ROOT / "metrics_summary.json"
DATA_QUALITY_REPORT = REPORTS_DIR / "data_quality_report.md"
DATA_DICTIONARY = REPORTS_DIR / "data_dictionary.md"
FEATURE_DICTIONARY = REPORTS_DIR / "feature_dictionary.md"

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
RANDOM_SEED = 42

# Reference epoch for converting ULB `Time` (seconds since first txn) to timestamps
TIME_EPOCH = "2013-09-01 00:00:00"