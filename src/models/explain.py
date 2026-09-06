"""
Feature importance and SHAP explainability for trained fraud models.

Produces:
  - Global feature importance (model-native or |coef|)
  - SHAP summary values on a sampled test set (TreeExplainer for RF/XGB,
    LinearExplainer for logistic regression)
  - CSV + PNG artifacts for the Streamlit dashboard
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.engineering import FEATURE_COLUMNS
from src.utils.paths import ARTIFACTS_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR, RANDOM_SEED


def _native_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
        kind = "feature_importances_"
    elif hasattr(model, "coef_"):
        vals = np.abs(np.asarray(model.coef_).ravel())
        kind = "abs_coef"
    else:
        raise ValueError("Model does not expose feature_importances_ or coef_")
    df = pd.DataFrame({"feature": feature_names, "importance": vals, "importance_type": kind})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    scaler: Any = None,
    max_samples: int = 1000,
    model_name: str = "model",
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Return mean |SHAP| per feature and raw shap matrix (samples x features).
    Falls back to native importance only if shap is unavailable.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = min(len(X), max_samples)
    idx = rng.choice(len(X), size=n, replace=False) if len(X) > n else np.arange(len(X))
    Xs = X.iloc[idx].copy()
    Xt = scaler.transform(Xs) if scaler is not None else Xs.to_numpy()
    if not isinstance(Xt, np.ndarray):
        Xt = np.asarray(Xt)
    Xt_df = pd.DataFrame(Xt, columns=list(X.columns))

    try:
        import shap
    except ImportError:
        native = _native_importance(model, list(X.columns))
        native["mean_abs_shap"] = native["importance"]
        return native[["feature", "mean_abs_shap"]], None

    # Prefer TreeExplainer for tree models; LinearExplainer for LR
    shap_matrix: Optional[np.ndarray] = None
    try:
        if model_name == "logistic_regression" or (hasattr(model, "coef_") and not hasattr(model, "feature_importances_")):
            # Background sample for linear explainer
            bg_n = min(100, len(Xt_df))
            background = shap.sample(Xt_df, bg_n, random_state=RANDOM_SEED)
            explainer = shap.LinearExplainer(model, background)
            sv = explainer.shap_values(Xt_df)
        else:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(Xt_df)

        if isinstance(sv, list):
            # binary classifiers sometimes return [class0, class1]
            shap_matrix = np.asarray(sv[1] if len(sv) > 1 else sv[0])
        else:
            shap_matrix = np.asarray(sv)
            if shap_matrix.ndim == 3:
                shap_matrix = shap_matrix[:, :, 1]
    except Exception:
        # Last resort: KernelExplainer is too slow; use native importance
        native = _native_importance(model, list(X.columns))
        native["mean_abs_shap"] = native["importance"]
        return native[["feature", "mean_abs_shap"]], None

    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    out = pd.DataFrame({"feature": list(X.columns), "mean_abs_shap": mean_abs})
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return out, shap_matrix


def plot_importance_bar(df: pd.DataFrame, value_col: str, title: str, out_path: Path, top_n: int = 20) -> Path:
    plot_df = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_df["feature"], plot_df[value_col], color="#264653")
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def explain_best_model(
    model_name: str,
    X_sample: pd.DataFrame,
    max_samples: int = 800,
) -> Dict[str, Any]:
    """Load saved model bundle and write importance + SHAP artifacts."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    bundle_path = MODELS_DIR / f"{model_name}.joblib"
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    scaler = bundle.get("scaler")
    features = bundle.get("features", FEATURE_COLUMNS)
    X = X_sample[features].copy()

    native = _native_importance(model, features)
    native_path = ARTIFACTS_DIR / "feature_importance.csv"
    native.to_csv(native_path, index=False)
    native_fig = plot_importance_bar(
        native.rename(columns={"importance": "importance"}),
        "importance",
        f"Native feature importance — {model_name}",
        FIGURES_DIR / "feature_importance.png",
    )

    shap_df, shap_matrix = compute_shap_values(
        model, X, scaler=scaler, max_samples=max_samples, model_name=model_name
    )
    shap_path = ARTIFACTS_DIR / "shap_mean_abs.csv"
    shap_df.to_csv(shap_path, index=False)
    shap_fig = plot_importance_bar(
        shap_df,
        "mean_abs_shap",
        f"Mean |SHAP| — {model_name}",
        FIGURES_DIR / "shap_summary.png",
    )

    summary = {
        "model_name": model_name,
        "n_samples_explained": int(min(len(X), max_samples)),
        "top_native": native.head(15).to_dict(orient="records"),
        "top_shap": shap_df.head(15).to_dict(orient="records"),
        "artifacts": {
            "feature_importance_csv": str(native_path),
            "feature_importance_png": str(native_fig),
            "shap_csv": str(shap_path),
            "shap_png": str(shap_fig),
            "shap_available": shap_matrix is not None,
        },
    }
    (METRICS_DIR / "explainability.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
