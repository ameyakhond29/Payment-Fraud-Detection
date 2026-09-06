"""Exploratory data analysis for the ULB credit-card fraud dataset."""

from __future__ import annotations

import json
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.paths import DATA_PROCESSED, FIGURES_DIR, METRICS_DIR


def run_eda(df: pd.DataFrame) -> Dict[str, Any]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "hour" not in df.columns:
        df["hour"] = ((df["seconds_from_start"] % 86400) // 3600).astype(int)
    if "amount_bin" not in df.columns:
        df["amount_bin"] = pd.cut(
            df["amount"].clip(lower=0),
            bins=[-0.01, 1, 25, 50, 100, 250, 500, 1000, 5000, 1e9],
            labels=["0-1", "1-25", "25-50", "50-100", "100-250", "250-500", "500-1000", "1000-5000", "5000+"],
        ).astype(str)

    summary: Dict[str, Any] = {
        "n_transactions": int(len(df)),
        "fraud_rate": float(df["is_fraud"].mean()),
        "n_fraud": int(df["is_fraud"].sum()),
        "n_legit": int((df["is_fraud"] == 0).sum()),
        "amount_stats": df["amount"].describe().to_dict(),
        "amount_stats_by_fraud": df.groupby("is_fraud")["amount"].describe().to_dict(),
        "note": "ULB dataset has no location/device/payment/merchant fields; EDA uses amount bins and hour.",
    }

    by_hour = (
        df.groupby("hour")
        .agg(transactions=("transaction_id", "count"), fraud=("is_fraud", "sum"), fraud_rate=("is_fraud", "mean"))
        .reset_index()
    )
    by_amount = (
        df.groupby("amount_bin")
        .agg(transactions=("transaction_id", "count"), fraud=("is_fraud", "sum"), fraud_rate=("is_fraud", "mean"),
             avg_amount=("amount", "mean"))
        .reset_index()
    )
    by_hour.to_csv(DATA_PROCESSED / "eda_fraud_by_hour.csv", index=False)
    by_amount.to_csv(DATA_PROCESSED / "eda_fraud_by_amount_bin.csv", index=False)

    # Proxy tables for dashboard compatibility naming
    by_amount.to_csv(DATA_PROCESSED / "eda_fraud_by_payment.csv", index=False)  # amount bins as stand-in analytics
    by_hour.to_csv(DATA_PROCESSED / "eda_fraud_by_location.csv", index=False)  # hour as temporal stand-in

    summary["fraud_by_hour"] = by_hour.to_dict(orient="records")
    summary["fraud_by_amount_bin"] = by_amount.to_dict(orient="records")

    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["is_fraud"].value_counts().sort_index()
    ax.bar(["Legitimate", "Fraud"], counts.values, color=["#2A9D8F", "#E76F51"])
    ax.set_title("Transaction Class Counts (ULB)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color in [(0, "#2A9D8F"), (1, "#E76F51")]:
        subset = df.loc[df["is_fraud"] == label, "amount"]
        ax.hist(
            subset.clip(upper=subset.quantile(0.99) if len(subset) else 0),
            bins=50,
            alpha=0.55,
            label="Fraud" if label else "Legitimate",
            color=color,
            density=True,
        )
    ax.set_title("Amount Distribution by Class (clipped at 99th pct)")
    ax.set_xlabel("Amount")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "amount_by_class.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(by_hour["hour"], by_hour["fraud_rate"], marker="o", color="#E76F51")
    ax.set_xlabel("Hour of Day (from Time)")
    ax.set_ylabel("Fraud Rate")
    ax.set_title("Fraud Rate by Hour")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fraud_by_hour.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    order = ["0-1", "1-25", "25-50", "50-100", "100-250", "250-500", "500-1000", "1000-5000", "5000+"]
    plot_df = by_amount.set_index("amount_bin").reindex([o for o in order if o in set(by_amount["amount_bin"])]).reset_index()
    ax.bar(plot_df["amount_bin"], plot_df["fraud_rate"], color="#264653")
    ax.set_ylabel("Fraud Rate")
    ax.set_title("Fraud Rate by Amount Bin")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fraud_by_amount_bin.png", dpi=120)
    # Also save under old names expected by docs
    fig.savefig(FIGURES_DIR / "fraud_by_payment.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(by_hour["hour"], by_hour["fraud_rate"], color="#F4A261")
    ax.set_title("Fraud Rate by Hour (bar)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fraud_by_location.png", dpi=120)
    fig.savefig(FIGURES_DIR / "fraud_by_device.png", dpi=120)
    plt.close(fig)

    # Daily-ish volume by hour buckets across stream
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["seconds_from_start"], range(len(df)), color="#264653", linewidth=0.5)
    ax.set_title("Cumulative Transactions over Time")
    ax.set_xlabel("Seconds from first transaction")
    ax.set_ylabel("Cumulative count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "daily_volume.png", dpi=120)
    plt.close(fig)

    out_path = METRICS_DIR / "eda_summary.json"
    out_path.write_text(json.dumps(json.loads(json.dumps(summary, default=str)), indent=2), encoding="utf-8")
    return summary
