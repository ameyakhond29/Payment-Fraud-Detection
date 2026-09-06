"""
Load and standardize the public ULB Credit Card Fraud dataset.

Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
File: creditcard.csv (Time, V1–V28, Amount, Class)

Note: This dataset does not contain location, device, payment method,
customer ID, or merchant ID — those fields are anonymized away via PCA.
The pipeline works with the available public fields and documents the gap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.utils.paths import (
    CREDITCARD_CSV,
    CREDITCARD_EXTERNAL,
    DATA_RAW,
    PCA_FEATURES,
    RAW_CSV,
    RAW_PARQUET,
    TIME_EPOCH,
)

PathLike = Union[str, Path]


def resolve_creditcard_path(path: Optional[PathLike] = None) -> Path:
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    for candidate in (CREDITCARD_CSV, CREDITCARD_EXTERNAL, DATA_RAW / "creditcard.csv"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "creditcard.csv not found. Place it at the project root or data/external/creditcard.csv"
    )


def load_creditcard(path: Optional[PathLike] = None) -> pd.DataFrame:
    """Load raw Kaggle/ULB creditcard.csv."""
    csv_path = resolve_creditcard_path(path)
    df = pd.read_csv(csv_path)
    expected = {"Time", "Amount", "Class", *PCA_FEATURES}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"creditcard.csv missing columns: {sorted(missing)}")
    return df


def standardize_creditcard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map ULB schema to project columns used downstream.

    - Class → is_fraud
    - Time → seconds_from_start + timestamp (derived)
    - Amount → amount
    - Add transaction_id
    - Keep V1–V28
    """
    out = df.copy()
    out["transaction_id"] = [f"T{i:08d}" for i in range(len(out))]
    out["seconds_from_start"] = pd.to_numeric(out["Time"], errors="coerce")
    out["amount"] = pd.to_numeric(out["Amount"], errors="coerce")
    out["is_fraud"] = pd.to_numeric(out["Class"], errors="coerce")
    out["timestamp"] = pd.to_datetime(TIME_EPOCH) + pd.to_timedelta(
        out["seconds_from_start"], unit="s"
    )
    # Amount bins for dashboard analytics (derived, documented)
    out["amount_bin"] = pd.cut(
        out["amount"].clip(lower=0),
        bins=[-0.01, 1, 25, 50, 100, 250, 500, 1000, 5000, 1e9],
        labels=[
            "0-1",
            "1-25",
            "25-50",
            "50-100",
            "100-250",
            "250-500",
            "500-1000",
            "1000-5000",
            "5000+",
        ],
    ).astype(str)
    out["hour"] = ((out["seconds_from_start"] % 86400) // 3600).astype("Int64")
    keep = [
        "transaction_id",
        "timestamp",
        "seconds_from_start",
        "amount",
        "amount_bin",
        "hour",
        "is_fraud",
    ] + PCA_FEATURES
    return out[keep]


def save_raw_dataset(df: pd.DataFrame) -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    # Raw standardized copy (still "raw" relative to cleaned/features)
    df.to_csv(RAW_CSV, index=False)
    df.to_parquet(RAW_PARQUET, index=False)


def ingest_creditcard(path: Optional[PathLike] = None) -> pd.DataFrame:
    """Load public CSV, standardize, and persist under data/raw/."""
    raw = load_creditcard(path)
    standardized = standardize_creditcard(raw)
    save_raw_dataset(standardized)
    return standardized
