"""
End-to-end pipeline on the public ULB creditcard.csv dataset:
  ingest → validate → EDA → features → train → metrics_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.eda import run_eda
from src.data.ingest import load_transactions, store_to_sqlite
from src.data.load_creditcard import ingest_creditcard, resolve_creditcard_path
from src.data.validate import (
    run_validation,
    save_clean_data,
    write_data_dictionary,
    write_quality_report,
)
from src.features.engineering import engineer_features, save_features, write_feature_dictionary
from src.models.train import train_all_models
from src.utils.paths import DB_PATH, METRICS_SUMMARY, RANDOM_SEED, REPORTS_DIR


def run_pipeline(creditcard_path: str | None = None, skip_ingest: bool = False) -> dict:
    t0 = time.time()
    print("=" * 60)
    print("Payment Fraud Detection — ULB creditcard.csv Pipeline")
    print("=" * 60)

    if not skip_ingest:
        src = resolve_creditcard_path(creditcard_path)
        print(f"\n[1/6] Ingesting public dataset: {src}")
        raw = ingest_creditcard(src)
    else:
        print("\n[1/6] Loading existing raw standardized dataset...")
        raw = load_transactions()
    print(f"      Raw rows: {len(raw):,} | Fraud rate: {raw['is_fraud'].mean():.4%}")

    write_data_dictionary()
    print("      Data dictionary written.")

    print("\n[2/6] Running data-quality validation (12 rules)...")
    clean, dq_report, _ = run_validation(raw)
    write_quality_report(dq_report)
    save_clean_data(clean)
    store_to_sqlite(clean, db_path=DB_PATH, table_name="transactions_clean", if_exists="replace")
    print(f"      Clean rows: {clean.shape[0]:,} | Removed: {dq_report.rows_removed:,}")

    print("\n[3/6] Running exploratory data analysis...")
    eda_summary = run_eda(clean)
    print(f"      Fraud rate: {eda_summary['fraud_rate']:.4%}")

    print("\n[4/6] Engineering features (leakage-safe)...")
    featured = engineer_features(clean)
    save_features(featured)
    write_feature_dictionary()
    store_to_sqlite(clean, db_path=DB_PATH, table_name="transactions_clean", if_exists="replace")
    # Features table can be wide; store slim analytics columns + key engineered fields
    feature_sql_cols = [
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
            "rolling_txn_rate_100",
        ]
        if c in featured.columns
    ]
    store_to_sqlite(
        featured[feature_sql_cols],
        db_path=DB_PATH,
        table_name="transactions_features",
        if_exists="replace",
    )
    print(f"      Feature rows: {len(featured):,} | Model features: {featured.shape[1]}")

    print("\n[5/6] Training Logistic Regression, Random Forest, XGBoost...")
    model_results = train_all_models(featured)
    best = model_results["best_model"]
    best_m = model_results["best_model_test_metrics"]
    print(f"      Best model: {best}")
    print(
        f"      Test Precision={best_m['precision']:.4f} Recall={best_m['recall']:.4f} "
        f"F1={best_m['f1']:.4f} ROC-AUC={best_m['roc_auc']:.4f} PR-AUC={best_m['pr_auc']:.4f}"
    )

    print("\n[6/6] Writing metrics_summary.json...")
    elapsed = time.time() - t0
    cost_aware = model_results.get("cost_aware", {})
    explainability = model_results.get("explainability", {})
    metrics_summary = {
        "dataset": {
            "source": "ULB Credit Card Fraud Detection (creditcard.csv)",
            "source_url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
            "n_raw_transactions": int(len(raw)),
            "n_clean_transactions": int(len(clean)),
            "n_rows_removed_by_dq": int(dq_report.rows_removed),
            "fraud_rate_clean": float(clean["is_fraud"].mean()),
            "n_features": int(model_results["split"]["feature_count"]),
            "schema_note": (
                "Public ULB fields only: Time, Amount, V1-V28, Class. "
                "Location/device/payment/customer/merchant IDs are not present and were not fabricated."
            ),
        },
        "data_quality": {
            "n_rules": dq_report.to_dict()["n_rules"],
            "n_rules_passed": dq_report.to_dict()["n_rules_passed"],
            "n_rules_failed": dq_report.to_dict()["n_rules_failed"],
            "rows_flagged": dq_report.rows_flagged,
            "rows_removed": dq_report.rows_removed,
            "rule_violations": {
                r.rule_id: {"name": r.name, "n_violations": r.n_violations, "passed": r.passed}
                for r in dq_report.rules
            },
        },
        "split": model_results["split"],
        "cost_config": model_results.get("cost_config", {}),
        "models": {
            name: {
                "precision": m["test"]["precision"],
                "recall": m["test"]["recall"],
                "f1": m["test"]["f1"],
                "roc_auc": m["test"]["roc_auc"],
                "pr_auc": m["test"]["pr_auc"],
                "confusion_matrix": m["test"]["confusion_matrix"],
                "false_positives": m["test"]["false_positives"],
                "threshold": m["test"]["threshold"],
                **(
                    {"best_params": m["best_params"], "best_val_pr_auc": m["best_val_pr_auc"]}
                    if name == "xgboost"
                    else {}
                ),
            }
            for name, m in model_results["models"].items()
        },
        "best_model": best,
        "best_model_test_metrics": {
            "precision": best_m["precision"],
            "recall": best_m["recall"],
            "f1": best_m["f1"],
            "roc_auc": best_m["roc_auc"],
            "pr_auc": best_m["pr_auc"],
            "confusion_matrix": best_m["confusion_matrix"],
            "false_positives": best_m["false_positives"],
            "threshold": best_m.get("threshold"),
        },
        "cost_aware": {
            "model": cost_aware.get("model"),
            "validation": cost_aware.get("validation"),
            "test": {
                "f1_threshold_cost": cost_aware.get("test", {}).get("f1_threshold_cost"),
                "cost_threshold_cost": cost_aware.get("test", {}).get("cost_threshold_cost"),
                "cost_threshold_metrics": {
                    k: cost_aware.get("test", {}).get("cost_threshold_metrics", {}).get(k)
                    for k in [
                        "precision",
                        "recall",
                        "f1",
                        "roc_auc",
                        "pr_auc",
                        "confusion_matrix",
                        "false_positives",
                        "false_negatives",
                        "threshold",
                    ]
                },
                "test_savings_vs_f1": cost_aware.get("test", {}).get("test_savings_vs_f1"),
            },
        },
        "explainability": {
            "model_name": explainability.get("model_name"),
            "n_samples_explained": explainability.get("n_samples_explained"),
            "top_shap": explainability.get("top_shap"),
            "top_native": explainability.get("top_native"),
            "artifacts": explainability.get("artifacts"),
        },
        "pipeline_runtime_seconds": round(elapsed, 2),
        "random_seed": RANDOM_SEED,
        "note": "All metrics calculated from actual pipeline execution on creditcard.csv. None are fabricated.",
    }
    METRICS_SUMMARY.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
    (REPORTS_DIR / "metrics" / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2), encoding="utf-8"
    )

    print(f"\nDone in {elapsed:.1f}s")
    print(f"metrics_summary.json -> {METRICS_SUMMARY}")
    return metrics_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fraud detection pipeline on creditcard.csv")
    parser.add_argument("--creditcard", type=str, default=None, help="Path to creditcard.csv")
    parser.add_argument("--skip-ingest", action="store_true", help="Reuse data/raw standardized files")
    args = parser.parse_args()
    run_pipeline(creditcard_path=args.creditcard, skip_ingest=args.skip_ingest)


if __name__ == "__main__":
    main()
