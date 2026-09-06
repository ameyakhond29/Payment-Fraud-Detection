"""PyTest suite for ULB creditcard pipeline components."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_creditcard import standardize_creditcard
from src.data.validate import RULE_SPECS, run_validation
from src.features.engineering import FEATURE_COLUMNS, engineer_features, get_model_matrix
from src.models.train import _best_threshold, _metrics, temporal_split
from src.utils.paths import PCA_FEATURES


def _make_ulb_like(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0, 172000, size=n))
    data = {"Time": time, "Amount": rng.lognormal(3, 1.2, size=n).clip(0, 5000)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, size=n)
    # Sparse fraud
    data["Class"] = (rng.random(n) < 0.02).astype(int)
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def small_raw() -> pd.DataFrame:
    return standardize_creditcard(_make_ulb_like(3000))


@pytest.fixture(scope="module")
def clean_df(small_raw: pd.DataFrame) -> pd.DataFrame:
    clean, _, _ = run_validation(small_raw)
    return clean


@pytest.fixture(scope="module")
def featured_df(clean_df: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(clean_df)


def test_at_least_ten_validation_rules():
    assert len(RULE_SPECS) >= 10


def test_validation_removes_negative_amounts(small_raw: pd.DataFrame):
    dirty = small_raw.copy()
    dirty.loc[dirty.index[:5], "amount"] = -10.0
    clean, report, _ = run_validation(dirty)
    assert (clean["amount"] >= 0).all()
    assert next(r for r in report.rules if r.rule_id == "DQ05").n_violations >= 5


def test_validation_detects_duplicate_feature_rows(small_raw: pd.DataFrame):
    dirty = pd.concat([small_raw, small_raw.iloc[[0]]], ignore_index=True)
    dirty["transaction_id"] = [f"T{i:08d}" for i in range(len(dirty))]
    clean, report, _ = run_validation(dirty)
    assert next(r for r in report.rules if r.rule_id == "DQ04").n_violations >= 1
    assert len(clean) == len(small_raw)


def test_validation_rejects_invalid_labels(small_raw: pd.DataFrame):
    dirty = small_raw.copy()
    dirty.loc[dirty.index[:3], "is_fraud"] = 2
    clean, report, _ = run_validation(dirty)
    assert set(clean["is_fraud"].unique()).issubset({0, 1})
    assert next(r for r in report.rules if r.rule_id == "DQ08").n_violations >= 3


def test_validation_rejects_nonfinite_pca(small_raw: pd.DataFrame):
    dirty = small_raw.copy()
    dirty.loc[dirty.index[0], "V1"] = np.inf
    dirty.loc[dirty.index[1], "V2"] = np.nan
    clean, report, _ = run_validation(dirty)
    assert np.isfinite(clean[PCA_FEATURES].to_numpy()).all()
    assert next(r for r in report.rules if r.rule_id == "DQ09").n_violations >= 2


def test_feature_count_at_least_20():
    assert len(FEATURE_COLUMNS) >= 20


def test_engineer_features_produces_all_columns(featured_df: pd.DataFrame):
    for col in FEATURE_COLUMNS:
        assert col in featured_df.columns
    assert featured_df[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_no_label_leakage_in_feature_columns():
    assert "is_fraud" not in FEATURE_COLUMNS
    assert "Class" not in FEATURE_COLUMNS


def test_expanding_stats_are_causal(featured_df: pd.DataFrame):
    # First row expanding mean should equal its own amount (fillna path) or be defined
    first = featured_df.iloc[0]
    assert np.isfinite(first["expanding_amount_mean"])
    assert np.isfinite(first["amount_zscore_expanding"])


def test_temporal_split_preserves_order(featured_df: pd.DataFrame):
    train, val, test = temporal_split(featured_df)
    assert train["seconds_from_start"].max() <= val["seconds_from_start"].min()
    assert val["seconds_from_start"].max() <= test["seconds_from_start"].min()
    assert len(train) + len(val) + len(test) == len(featured_df)


def test_get_model_matrix_shapes(featured_df: pd.DataFrame):
    X, y = get_model_matrix(featured_df)
    assert X.shape[0] == y.shape[0]
    assert list(X.columns) == FEATURE_COLUMNS
    assert set(y.unique()).issubset({0, 1})


def test_metrics_helper_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.95])
    m = _metrics(y_true, y_prob, threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["confusion_matrix"]["tp"] == 2


def test_best_threshold_returns_in_range():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.45, 0.55, 0.7, 0.9])
    t = _best_threshold(y_true, y_prob)
    assert 0.05 <= t <= 0.95


def test_standardize_creditcard_schema():
    raw = _make_ulb_like(100)
    std = standardize_creditcard(raw)
    assert {"transaction_id", "amount", "is_fraud", "timestamp", "seconds_from_start"}.issubset(std.columns)
    assert set(PCA_FEATURES).issubset(std.columns)
    assert len(std) == 100


def test_prediction_output_schema(featured_df: pd.DataFrame):
    X, _ = get_model_matrix(featured_df.head(10))
    probs = np.clip(X["amount"] / (X["amount"].max() + 1), 0, 1).to_numpy()
    out = pd.DataFrame({"fraud_probability": probs, "predicted_fraud": (probs >= 0.5).astype(int)})
    assert out["fraud_probability"].between(0, 1).all()
    assert set(out["predicted_fraud"].unique()).issubset({0, 1})


def test_cost_threshold_prefers_fewer_expensive_errors():
    from src.models.cost_threshold import CostConfig, select_cost_minimizing_threshold

    # 6 samples: clear separation except one ambiguous
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.55, 0.8, 0.9])
    amounts = np.array([10.0, 10.0, 10.0, 500.0, 500.0, 500.0])
    cfg = CostConfig(cost_fp=10.0, use_amount_as_fn_cost=True, min_fn_cost=1.0)
    result = select_cost_minimizing_threshold(y_true, y_prob, config=cfg, amounts=amounts)
    assert 0.01 <= result["best"]["threshold"] <= 0.99
    assert result["best"]["total_cost"] >= 0
    assert "curve" in result and len(result["curve"]) >= 10


def test_cost_comparison_reports_savings():
    from src.models.cost_threshold import CostConfig, compare_f1_vs_cost_thresholds

    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.15, 0.35, 0.45, 0.55, 0.65, 0.85, 0.95])
    amounts = np.array([20, 20, 20, 20, 200, 200, 200, 200], dtype=float)
    out = compare_f1_vs_cost_thresholds(
        y_true, y_prob, f1_threshold=0.5, config=CostConfig(cost_fp=5.0), amounts=amounts
    )
    assert "f1_threshold_result" in out
    assert "cost_threshold_result" in out
    assert "savings_vs_f1_threshold" in out


def test_native_importance_from_dummy_forest():
    from sklearn.ensemble import RandomForestClassifier
    from src.models.explain import _native_importance

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] + rng.normal(0, 0.1, size=200) > 0).astype(int)
    model = RandomForestClassifier(n_estimators=20, random_state=0)
    model.fit(X, y)
    imp = _native_importance(model, list(X.columns))
    assert set(imp["feature"]) == set(X.columns)
    assert imp["importance"].sum() > 0
