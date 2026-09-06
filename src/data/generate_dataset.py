"""
Synthetic payment transaction dataset generator.

Produces >= 100,000 transactions with realistic fraud patterns.
Fraud labels are assigned using documented behavioral rules so the
dataset is learnable without fabricating evaluation metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import (
    DATA_RAW,
    DEVICE_TYPES,
    FRAUD_RATE,
    LOCATIONS,
    N_TRANSACTIONS,
    PAYMENT_METHODS,
    RANDOM_SEED,
    RAW_CSV,
    RAW_PARQUET,
)


def _customer_profiles(n_customers: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate stable customer behavioral baselines."""
    home_idx = rng.integers(0, len(LOCATIONS), size=n_customers)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n_customers)],
            "home_location": [LOCATIONS[i] for i in home_idx],
            "preferred_device": rng.choice(DEVICE_TYPES, size=n_customers),
            "preferred_payment": rng.choice(PAYMENT_METHODS, size=n_customers),
            "avg_amount": rng.lognormal(mean=3.8, sigma=0.7, size=n_customers).clip(5, 2500),
            "txn_rate_per_day": rng.uniform(0.2, 4.0, size=n_customers),
            "risk_score": rng.beta(a=1.5, b=8.0, size=n_customers),
        }
    )


def generate_transactions(
    n_transactions: int = N_TRANSACTIONS,
    seed: int = RANDOM_SEED,
    fraud_rate: float = FRAUD_RATE,
) -> pd.DataFrame:
    """
    Generate a synthetic transaction dataset.

    Fraud is injected via:
    - High amount relative to customer average
    - Location mismatch vs home location
    - Unusual device / payment method
    - Burst velocity (many txns in short window) encoded via hour patterns
    - Elevated customer risk score
    Plus a small random fraud noise component.
    """
    rng = np.random.default_rng(seed)

    n_customers = max(2_000, n_transactions // 40)
    n_merchants = max(500, n_transactions // 200)

    customers = _customer_profiles(n_customers, rng)
    merchant_ids = [f"M{i:05d}" for i in range(n_merchants)]
    merchant_risk = {m: float(rng.beta(2, 10)) for m in merchant_ids}

    # Time range: 180 days ending 2024-12-31
    end = pd.Timestamp("2024-12-31 23:59:59")
    start = end - pd.Timedelta(days=180)
    timestamps = pd.to_datetime(
        rng.integers(int(start.timestamp()), int(end.timestamp()), size=n_transactions),
        unit="s",
    )

    cust_idx = rng.integers(0, n_customers, size=n_transactions)
    cust = customers.iloc[cust_idx].reset_index(drop=True)

    # Amounts skewed lognormal around customer average
    amount_noise = rng.lognormal(mean=0.0, sigma=0.55, size=n_transactions)
    amounts = (cust["avg_amount"].to_numpy() * amount_noise).clip(0.5, 15_000)

    # Mostly home location; ~18% travel / mismatch
    locations = cust["home_location"].to_numpy().copy()
    mismatch_mask = rng.random(n_transactions) < 0.18
    locations[mismatch_mask] = rng.choice(LOCATIONS, size=mismatch_mask.sum())

    devices = cust["preferred_device"].to_numpy().copy()
    device_switch = rng.random(n_transactions) < 0.15
    devices[device_switch] = rng.choice(DEVICE_TYPES, size=device_switch.sum())

    payments = cust["preferred_payment"].to_numpy().copy()
    pay_switch = rng.random(n_transactions) < 0.12
    payments[pay_switch] = rng.choice(PAYMENT_METHODS, size=pay_switch.sum())

    merchants = rng.choice(merchant_ids, size=n_transactions)
    m_risk = np.array([merchant_risk[m] for m in merchants])

    hour = timestamps.hour.to_numpy() if hasattr(timestamps, "hour") else pd.Series(timestamps).dt.hour.to_numpy()
    night = ((hour >= 0) & (hour <= 5)).astype(float)

    # Fraud score (higher => more likely fraud)
    amt_ratio = amounts / np.maximum(cust["avg_amount"].to_numpy(), 1.0)
    loc_mismatch = (locations != cust["home_location"].to_numpy()).astype(float)
    device_mismatch = (devices != cust["preferred_device"].to_numpy()).astype(float)
    pay_mismatch = (payments != cust["preferred_payment"].to_numpy()).astype(float)
    high_amount = (amt_ratio > 3.0).astype(float)
    extreme_amount = (amt_ratio > 6.0).astype(float)
    crypto = (payments == "crypto").astype(float)

    fraud_logit = (
        -4.2
        + 1.4 * high_amount
        + 1.8 * extreme_amount
        + 1.1 * loc_mismatch
        + 0.7 * device_mismatch
        + 0.6 * pay_mismatch
        + 0.9 * night
        + 1.5 * cust["risk_score"].to_numpy()
        + 1.2 * m_risk
        + 0.8 * crypto
        + rng.normal(0, 0.35, size=n_transactions)
    )
    fraud_prob = 1.0 / (1.0 + np.exp(-fraud_logit))

    # Calibrate toward target fraud rate via threshold on probability ranks
    threshold = np.quantile(fraud_prob, 1.0 - fraud_rate)
    is_fraud = (fraud_prob >= threshold).astype(int)

    # Inject a few quality issues for the validation layer to catch (~0.8% dirty rows)
    dirty_n = int(n_transactions * 0.008)
    dirty_idx = rng.choice(n_transactions, size=dirty_n, replace=False)

    transaction_ids = [f"T{i:08d}" for i in range(n_transactions)]

    df = pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "customer_id": cust["customer_id"].to_numpy(),
            "merchant_id": merchants,
            "amount": np.round(amounts, 2),
            "timestamp": timestamps,
            "location": locations,
            "device_type": devices,
            "payment_method": payments,
            "is_fraud": is_fraud,
        }
    )

    # Intentionally inject data-quality issues into a small subset
    if dirty_n > 0:
        # Missing values
        miss = dirty_idx[: dirty_n // 5]
        for col in ["location", "device_type", "payment_method"]:
            if len(miss):
                df.loc[miss[: len(miss) // 3], col] = np.nan

        # Invalid amounts (negative / zero)
        inv_amt = dirty_idx[dirty_n // 5 : 2 * dirty_n // 5]
        df.loc[inv_amt[: len(inv_amt) // 2], "amount"] = -abs(float(rng.uniform(1, 100)))
        df.loc[inv_amt[len(inv_amt) // 2 :], "amount"] = 0.0

        # Invalid categories
        inv_cat = dirty_idx[2 * dirty_n // 5 : 3 * dirty_n // 5]
        if len(inv_cat):
            df.loc[inv_cat[: len(inv_cat) // 2], "device_type"] = "smartwatch_x"
            df.loc[inv_cat[len(inv_cat) // 2 :], "payment_method"] = "barter"

        # Invalid IDs
        inv_id = dirty_idx[3 * dirty_n // 5 : 4 * dirty_n // 5]
        if len(inv_id):
            df.loc[inv_id[: len(inv_id) // 2], "customer_id"] = ""
            df.loc[inv_id[len(inv_id) // 2 :], "merchant_id"] = "INVALID"

        # Invalid labels
        inv_lab = dirty_idx[4 * dirty_n // 5 :]
        if len(inv_lab):
            df.loc[inv_lab[: len(inv_lab) // 2], "is_fraud"] = 2
            df.loc[inv_lab[len(inv_lab) // 2 :], "is_fraud"] = -1

        # Duplicate transaction IDs (append clones)
        dup_src = dirty_idx[: min(40, dirty_n)]
        duplicates = df.iloc[dup_src].copy()
        duplicates["amount"] = duplicates["amount"] * 1.01  # slight difference
        df = pd.concat([df, duplicates], ignore_index=True)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def save_raw_dataset(df: pd.DataFrame, csv_path: Path = RAW_CSV, parquet_path: Path = RAW_PARQUET) -> None:
    """Persist raw data separately from processed outputs."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic fraud transaction dataset")
    parser.add_argument("--n", type=int, default=N_TRANSACTIONS, help="Number of transactions")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--fraud-rate", type=float, default=FRAUD_RATE)
    args = parser.parse_args()

    print(f"Generating {args.n:,} transactions (seed={args.seed})...")
    df = generate_transactions(n_transactions=args.n, seed=args.seed, fraud_rate=args.fraud_rate)
    save_raw_dataset(df)
    fraud_pct = 100.0 * (df["is_fraud"] == 1).mean()
    print(f"Saved raw CSV -> {RAW_CSV}")
    print(f"Saved raw Parquet -> {RAW_PARQUET}")
    print(f"Rows: {len(df):,} | Fraud label==1 rate: {fraud_pct:.3f}%")


if __name__ == "__main__":
    main()