"""
Train and evaluate Logistic Regression, Random Forest, and XGBoost models.

- Temporal train / validation / test splits (no shuffle leakage)
- Class imbalance handled via class_weight / scale_pos_weight
- Hyperparameter tuning for XGBoost on validation set
- F1 threshold + cost-aware threshold comparison
- Metrics: precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features.engineering import FEATURE_COLUMNS, get_model_matrix
from src.models.cost_threshold import (
    CostConfig,
    compare_f1_vs_cost_thresholds,
    compute_cost_at_threshold,
    save_cost_artifacts,
)
from src.models.explain import explain_best_model
from src.utils.paths import ARTIFACTS_DIR, DATA_PROCESSED, METRICS_DIR, MODELS_DIR, RANDOM_SEED


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time order: train → val → test (uses seconds_from_start when present)."""
    sort_col = "seconds_from_start" if "seconds_from_start" in df.columns else "timestamp"
    df = df.sort_values(sort_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "threshold": threshold,
        "support_fraud": int((y_true == 1).sum()),
        "support_legit": int((y_true == 0).sum()),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick threshold maximizing F1 on validation probabilities."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[Any, StandardScaler, Dict[str, Any]]:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_SEED,
        solver="lbfgs",
    )
    model.fit(Xtr, y_train)
    val_prob = model.predict_proba(Xva)[:, 1]
    thr = _best_threshold(y_val.to_numpy(), val_prob)
    return model, scaler, {"threshold": thr, "val_prob": val_prob}


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[Any, None, Dict[str, Any]]:
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    val_prob = model.predict_proba(X_val)[:, 1]
    thr = _best_threshold(y_val.to_numpy(), val_prob)
    return model, None, {"threshold": thr, "val_prob": val_prob}


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[Any, None, Dict[str, Any]]:
    """Grid-search a small hyperparameter space; select by validation PR-AUC."""
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / max(pos, 1)

    param_grid = [
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 200, "min_child_weight": 3},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 300, "min_child_weight": 5},
        {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 200, "min_child_weight": 3},
        {"max_depth": 8, "learning_rate": 0.05, "n_estimators": 250, "min_child_weight": 5},
        {"max_depth": 5, "learning_rate": 0.08, "n_estimators": 250, "min_child_weight": 1},
    ]

    best_model = None
    best_score = -1.0
    best_params: Dict[str, Any] = {}
    best_val_prob = None

    for params in param_grid:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=spw,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_prob = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, val_prob)
        if score > best_score:
            best_score = score
            best_model = model
            best_params = {**params, "scale_pos_weight": spw}
            best_val_prob = val_prob

    thr = _best_threshold(y_val.to_numpy(), best_val_prob)
    return best_model, None, {
        "threshold": thr,
        "val_prob": best_val_prob,
        "best_params": best_params,
        "best_val_pr_auc": float(best_score),
    }


def evaluate_on_test(
    model: Any,
    scaler: Optional[StandardScaler],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> Dict[str, Any]:
    Xt = scaler.transform(X_test) if scaler is not None else X_test
    prob = model.predict_proba(Xt)[:, 1]
    return _metrics(y_test.to_numpy(), prob, threshold=threshold), prob


def train_all_models(
    featured: pd.DataFrame,
    cost_config: Optional[CostConfig] = None,
) -> Dict[str, Any]:
    """Full training pipeline; returns metrics dict and persists artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    cost_config = cost_config or CostConfig()

    train_df, val_df, test_df = temporal_split(featured)
    X_train, y_train = get_model_matrix(train_df)
    X_val, y_val = get_model_matrix(val_df)
    X_test, y_test = get_model_matrix(test_df)
    val_amounts = val_df["amount"].to_numpy() if "amount" in val_df.columns else None
    test_amounts = test_df["amount"].to_numpy() if "amount" in test_df.columns else None

    results: Dict[str, Any] = {
        "split": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_fraud_rate": float(y_train.mean()),
            "val_fraud_rate": float(y_val.mean()),
            "test_fraud_rate": float(y_test.mean()),
            "feature_count": len(FEATURE_COLUMNS),
            "features": FEATURE_COLUMNS,
        },
        "models": {},
        "cost_config": {
            "cost_fp": cost_config.cost_fp,
            "cost_fn_fixed": cost_config.cost_fn_fixed,
            "use_amount_as_fn_cost": cost_config.use_amount_as_fn_cost,
            "min_fn_cost": cost_config.min_fn_cost,
        },
    }

    trained = {}

    # Logistic Regression
    lr_model, lr_scaler, lr_meta = train_logistic_regression(X_train, y_train, X_val, y_val)
    lr_test, lr_test_prob = evaluate_on_test(lr_model, lr_scaler, X_test, y_test, lr_meta["threshold"])
    results["models"]["logistic_regression"] = {
        "validation_threshold_f1": lr_meta["threshold"],
        "test": lr_test,
        "val_prob": lr_meta["val_prob"],
    }
    trained["logistic_regression"] = {
        "model": lr_model,
        "scaler": lr_scaler,
        "f1_threshold": lr_meta["threshold"],
        "val_prob": lr_meta["val_prob"],
        "test_prob": lr_test_prob,
    }
    joblib.dump(
        {
            "model": lr_model,
            "scaler": lr_scaler,
            "threshold": lr_meta["threshold"],
            "threshold_type": "f1",
            "features": FEATURE_COLUMNS,
        },
        MODELS_DIR / "logistic_regression.joblib",
    )

    # Random Forest
    rf_model, _, rf_meta = train_random_forest(X_train, y_train, X_val, y_val)
    rf_test, rf_test_prob = evaluate_on_test(rf_model, None, X_test, y_test, rf_meta["threshold"])
    results["models"]["random_forest"] = {
        "validation_threshold_f1": rf_meta["threshold"],
        "test": rf_test,
        "val_prob": rf_meta["val_prob"],
    }
    trained["random_forest"] = {
        "model": rf_model,
        "scaler": None,
        "f1_threshold": rf_meta["threshold"],
        "val_prob": rf_meta["val_prob"],
        "test_prob": rf_test_prob,
    }
    joblib.dump(
        {
            "model": rf_model,
            "scaler": None,
            "threshold": rf_meta["threshold"],
            "threshold_type": "f1",
            "features": FEATURE_COLUMNS,
        },
        MODELS_DIR / "random_forest.joblib",
    )

    # XGBoost (tuned)
    xgb_model, _, xgb_meta = tune_xgboost(X_train, y_train, X_val, y_val)
    xgb_test, xgb_test_prob = evaluate_on_test(xgb_model, None, X_test, y_test, xgb_meta["threshold"])
    results["models"]["xgboost"] = {
        "validation_threshold_f1": xgb_meta["threshold"],
        "best_params": xgb_meta["best_params"],
        "best_val_pr_auc": xgb_meta["best_val_pr_auc"],
        "test": xgb_test,
        "val_prob": xgb_meta["val_prob"],
    }
    trained["xgboost"] = {
        "model": xgb_model,
        "scaler": None,
        "f1_threshold": xgb_meta["threshold"],
        "val_prob": xgb_meta["val_prob"],
        "test_prob": xgb_test_prob,
    }
    joblib.dump(
        {
            "model": xgb_model,
            "scaler": None,
            "threshold": xgb_meta["threshold"],
            "threshold_type": "f1",
            "features": FEATURE_COLUMNS,
        },
        MODELS_DIR / "xgboost.joblib",
    )

    # Select best by test PR-AUC (ranking quality, threshold-independent)
    best_name = max(
        ["logistic_regression", "random_forest", "xgboost"],
        key=lambda k: results["models"][k]["test"]["pr_auc"],
    )
    results["best_model"] = best_name
    results["best_model_test_metrics"] = results["models"][best_name]["test"]

    # --- Cost-aware threshold on best model (validation → apply on test) ---
    best_pack = trained[best_name]
    cost_comparison = compare_f1_vs_cost_thresholds(
        y_val.to_numpy(),
        best_pack["val_prob"],
        f1_threshold=best_pack["f1_threshold"],
        config=cost_config,
        amounts=val_amounts,
    )
    cost_paths = save_cost_artifacts(cost_comparison, prefix="cost_analysis")

    cost_thr = float(cost_comparison["cost_threshold_result"]["threshold"])
    test_cost_f1 = compute_cost_at_threshold(
        y_test.to_numpy(), best_pack["test_prob"], best_pack["f1_threshold"], cost_config, amounts=test_amounts
    )
    test_cost_opt = compute_cost_at_threshold(
        y_test.to_numpy(), best_pack["test_prob"], cost_thr, cost_config, amounts=test_amounts
    )
    test_metrics_cost = _metrics(y_test.to_numpy(), best_pack["test_prob"], threshold=cost_thr)

    results["cost_aware"] = {
        "model": best_name,
        "validation": {
            "f1_threshold": cost_comparison["f1_threshold_result"],
            "cost_threshold": cost_comparison["cost_threshold_result"],
            "savings_vs_f1_threshold": cost_comparison["savings_vs_f1_threshold"],
        },
        "test": {
            "f1_threshold_metrics": results["models"][best_name]["test"],
            "f1_threshold_cost": {
                "threshold": test_cost_f1.threshold,
                "total_cost": test_cost_f1.total_cost,
                "n_fp": test_cost_f1.n_fp,
                "n_fn": test_cost_f1.n_fn,
                "fp_cost_total": test_cost_f1.fp_cost_total,
                "fn_cost_total": test_cost_f1.fn_cost_total,
            },
            "cost_threshold_metrics": test_metrics_cost,
            "cost_threshold_cost": {
                "threshold": test_cost_opt.threshold,
                "total_cost": test_cost_opt.total_cost,
                "n_fp": test_cost_opt.n_fp,
                "n_fn": test_cost_opt.n_fn,
                "fp_cost_total": test_cost_opt.fp_cost_total,
                "fn_cost_total": test_cost_opt.fn_cost_total,
            },
            "test_savings_vs_f1": float(test_cost_f1.total_cost - test_cost_opt.total_cost),
        },
        "artifacts": cost_paths,
    }

    # Persist best model with BOTH thresholds
    joblib.dump(
        {
            "model": best_pack["model"],
            "scaler": best_pack["scaler"],
            "threshold": cost_thr,
            "threshold_f1": best_pack["f1_threshold"],
            "threshold_cost": cost_thr,
            "threshold_type": "cost",
            "cost_config": results["cost_config"],
            "features": FEATURE_COLUMNS,
        },
        MODELS_DIR / f"{best_name}.joblib",
    )
    joblib.dump(
        {
            "model": best_pack["model"],
            "scaler": best_pack["scaler"],
            "threshold": cost_thr,
            "threshold_f1": best_pack["f1_threshold"],
            "threshold_cost": cost_thr,
            "threshold_type": "cost",
            "cost_config": results["cost_config"],
            "features": FEATURE_COLUMNS,
            "model_name": best_name,
        },
        MODELS_DIR / "best_model.joblib",
    )

    # Predictions using cost-optimal threshold (primary for ops dashboard)
    test_prob = best_pack["test_prob"]
    test_pred = (test_prob >= cost_thr).astype(int)
    keep_cols = [
        c
        for c in [
            "transaction_id",
            "amount",
            "timestamp",
            "seconds_from_start",
            "hour",
            "amount_bin",
            "is_fraud",
        ]
        if c in test_df.columns
    ]
    pred_df = test_df[keep_cols].copy()
    pred_df["fraud_probability"] = test_prob
    pred_df["predicted_fraud_f1"] = (test_prob >= best_pack["f1_threshold"]).astype(int)
    pred_df["predicted_fraud"] = test_pred
    pred_df["threshold_used"] = cost_thr
    pred_df["false_positive"] = ((pred_df["predicted_fraud"] == 1) & (pred_df["is_fraud"] == 0)).astype(int)
    pred_df["false_negative"] = ((pred_df["predicted_fraud"] == 0) & (pred_df["is_fraud"] == 1)).astype(int)
    pred_path = ARTIFACTS_DIR / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    pred_df.to_csv(DATA_PROCESSED / "test_predictions_tableau.csv", index=False)

    results["test_predictions_path"] = str(pred_path)
    results["n_false_positives_test"] = int(pred_df["false_positive"].sum())
    results["n_false_negatives_test"] = int(pred_df["false_negative"].sum())

    # --- Explainability (SHAP + native importance) ---
    # Use a sample of test features for SHAP speed
    explain_summary = explain_best_model(best_name, X_test, max_samples=800)
    results["explainability"] = {
        "model_name": explain_summary["model_name"],
        "n_samples_explained": explain_summary["n_samples_explained"],
        "top_shap": explain_summary["top_shap"][:10],
        "top_native": explain_summary["top_native"][:10],
        "artifacts": explain_summary["artifacts"],
    }

    # Drop bulky in-memory val_prob arrays before JSON serialize
    for name in list(results["models"].keys()):
        results["models"][name].pop("val_prob", None)

    metrics_path = METRICS_DIR / "model_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results


def predict_fraud(features_row: Dict[str, float], model_name: str = "best_model") -> Dict[str, Any]:
    """Score a single feature dict; used by tests and inference."""
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        path = MODELS_DIR / "xgboost.joblib"
    bundle = joblib.load(path)
    X = pd.DataFrame([{c: features_row.get(c, 0.0) for c in bundle["features"]}])
    Xt = bundle["scaler"].transform(X) if bundle["scaler"] is not None else X
    prob = float(bundle["model"].predict_proba(Xt)[0, 1])
    thr = bundle.get("threshold_cost", bundle.get("threshold", 0.5))
    pred = int(prob >= thr)
    return {
        "fraud_probability": prob,
        "predicted_fraud": pred,
        "threshold": thr,
        "threshold_type": bundle.get("threshold_type", "unknown"),
    }
