"""
Feature engineering for the ULB credit-card fraud dataset.

Creates 20+ features from Amount, Time, and V1–V28 with leakage controls:
- Expanding / shifted stats never include the current row's label
- `is_fraud` / `Class` are never used as inputs
- Temporal ordering by `seconds_from_start` before any rolling logic
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.paths import DATA_PROCESSED, FEATURES_CSV, FEATURES_PARQUET, FEATURE_DICTIONARY, PCA_FEATURES

FEATURE_COLUMNS: List[str] = (
    [
        "amount",
        "log_amount",
        "amount_sqrt",
        "hour",
        "hour_sin",
        "hour_cos",
        "day_index",
        "seconds_from_start",
        "seconds_since_prev",
        "amount_delta_prev",
        "expanding_amount_mean",
        "expanding_amount_std",
        "amount_zscore_expanding",
        "rolling_amount_mean_100",
        "rolling_amount_std_100",
        "rolling_txn_rate_100",
        "pca_l2_norm",
        "pca_abs_max",
        "pca_abs_mean",
        "pca_abs_std",
    ]
    + PCA_FEATURES
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["seconds_from_start"] = pd.to_numeric(out["seconds_from_start"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out.sort_values(["seconds_from_start", "transaction_id"]).reset_index(drop=True)

    out["log_amount"] = np.log1p(out["amount"].clip(lower=0))
    out["amount_sqrt"] = np.sqrt(out["amount"].clip(lower=0))
    out["hour"] = ((out["seconds_from_start"] % 86400) // 3600).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["day_index"] = (out["seconds_from_start"] // 86400).astype(int)

    # Sequence / velocity (dataset has no customer_id; use global ordered stream)
    out["seconds_since_prev"] = out["seconds_from_start"].diff().fillna(0.0)
    out["amount_delta_prev"] = out["amount"].diff().fillna(0.0)

    # Past-only expanding amount stats (shift so current amount excluded from mean/std used in zscore base)
    shifted_amt = out["amount"].shift(1)
    out["expanding_amount_mean"] = shifted_amt.expanding(min_periods=1).mean().fillna(out["amount"])
    out["expanding_amount_std"] = shifted_amt.expanding(min_periods=2).std().fillna(0.0)
    out["amount_zscore_expanding"] = (
        (out["amount"] - out["expanding_amount_mean"])
        / out["expanding_amount_std"].replace(0, np.nan)
    ).fillna(0.0)

    # Rolling windows on prior rows only
    prior_amt = out["amount"].shift(1)
    out["rolling_amount_mean_100"] = prior_amt.rolling(100, min_periods=1).mean().fillna(out["amount"])
    out["rolling_amount_std_100"] = prior_amt.rolling(100, min_periods=2).std().fillna(0.0)
    # Approximate recent velocity: inverse of mean inter-arrival in last 100 gaps
    prior_gap = out["seconds_since_prev"].shift(1)
    mean_gap = prior_gap.rolling(100, min_periods=1).mean().replace(0, np.nan)
    out["rolling_txn_rate_100"] = (1.0 / mean_gap).fillna(0.0)

    pca = out[PCA_FEATURES].to_numpy(dtype=float)
    out["pca_l2_norm"] = np.linalg.norm(pca, axis=1)
    out["pca_abs_max"] = np.max(np.abs(pca), axis=1)
    out["pca_abs_mean"] = np.mean(np.abs(pca), axis=1)
    out["pca_abs_std"] = np.std(pca, axis=1)

    for col in FEATURE_COLUMNS:
        if col not in out.columns:
            raise KeyError(f"Missing engineered feature: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out


def get_model_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    y = df["is_fraud"].astype(int)
    return X, y


def save_features(df: pd.DataFrame, csv_path: Path = FEATURES_CSV, parquet_path: Path = FEATURES_PARQUET) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    slim_cols = [
        c
        for c in [
            "transaction_id",
            "timestamp",
            "seconds_from_start",
            "amount",
            "amount_bin",
            "hour",
            "is_fraud",
            "log_amount",
            "pca_l2_norm",
            "amount_zscore_expanding",
            "seconds_since_prev",
        ]
        if c in df.columns
    ]
    # Tableau-compatible slim CSV + full feature parquet
    df[slim_cols].to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)


def write_feature_dictionary(path: Optional[Path] = None) -> Path:
    path = path or FEATURE_DICTIONARY
    rows = [
        ("amount", "Transaction amount (EUR)", "Transaction", "Raw public field"),
        ("log_amount", "log(1 + amount)", "Transaction", "No label used"),
        ("amount_sqrt", "sqrt(amount)", "Transaction", "No label used"),
        ("hour", "Hour of day from Time", "Temporal", "Derived from Time"),
        ("hour_sin / hour_cos", "Cyclical hour encoding", "Temporal", "No label used"),
        ("day_index", "Day index from Time // 86400", "Temporal", "No label used"),
        ("seconds_from_start", "ULB Time field", "Temporal", "Raw public field"),
        ("seconds_since_prev", "Gap vs previous transaction", "Velocity", "diff / past-only"),
        ("amount_delta_prev", "Amount change vs previous txn", "Velocity", "diff / past-only"),
        ("expanding_amount_mean", "Expanding mean of prior amounts", "Behavior", "shift+expanding"),
        ("expanding_amount_std", "Expanding std of prior amounts", "Behavior", "shift+expanding"),
        ("amount_zscore_expanding", "Amount z-score vs prior stream", "Behavior", "past stats only"),
        ("rolling_amount_mean_100", "Rolling mean of prior 100 amounts", "Behavior", "shift+rolling"),
        ("rolling_amount_std_100", "Rolling std of prior 100 amounts", "Behavior", "shift+rolling"),
        ("rolling_txn_rate_100", "Approx rate from prior inter-arrival", "Velocity", "past gaps only"),
        ("pca_l2_norm", "L2 norm of V1–V28", "PCA aggregate", "No label used"),
        ("pca_abs_max / mean / std", "Abs stats over PCA vector", "PCA aggregate", "No label used"),
        ("V1–V28", "Anonymized PCA components from ULB", "PCA", "Public features; not labels"),
    ]
    lines = [
        "# Feature Dictionary",
        "",
        f"Total model features: **{len(FEATURE_COLUMNS)}** "
        f"(20 engineered + aggregates, including V1–V28).",
        "",
        "Leakage controls: expanding/rolling amount statistics use `shift(1)` so the",
        "current row is excluded. `is_fraud` / `Class` are never model inputs.",
        "",
        "Note: ULB data has no customer/merchant IDs, so behavioral features are",
        "computed on the globally time-ordered transaction stream.",
        "",
        "| Feature | Description | Category | Leakage control |",
        "|---------|-------------|----------|-----------------|",
    ]
    for name, desc, cat, leak in rows:
        lines.append(f"| `{name}` | {desc} | {cat} | {leak} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
