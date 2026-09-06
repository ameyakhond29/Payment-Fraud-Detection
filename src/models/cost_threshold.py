"""
Cost-aware decision thresholds for fraud scoring.

Business cost model (configurable):
  - FP cost: reviewing / blocking a legitimate payment
  - FN cost: missing fraud (default = expected loss ≈ transaction amount,
    or a fixed average fraud loss when amount is unavailable)

Total expected cost on a labeled set:
  cost = n_fp * cost_fp + sum(loss for each missed fraud)

Threshold is chosen on the validation set to minimize total cost, then
frozen for test evaluation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from src.utils.paths import ARTIFACTS_DIR, FIGURES_DIR, METRICS_DIR


ArrayLike = Union[np.ndarray, pd.Series, Sequence[float]]


@dataclass
class CostConfig:
    """Dollar costs used for threshold selection."""

    cost_fp: float = 10.0  # cost of a false positive (review / friction)
    cost_fn_fixed: float = 100.0  # fallback FN cost when amount unknown
    use_amount_as_fn_cost: bool = True  # FN cost = amount of missed fraud txn
    min_fn_cost: float = 1.0  # floor so zero-amount fraud still costs something

    def fn_cost_for_amounts(self, amounts: Optional[ArrayLike], n: int) -> np.ndarray:
        if self.use_amount_as_fn_cost and amounts is not None:
            amt = np.asarray(amounts, dtype=float)
            if len(amt) != n:
                raise ValueError("amounts length must match y_true")
            return np.maximum(amt, self.min_fn_cost)
        return np.full(n, self.cost_fn_fixed, dtype=float)


@dataclass
class ThresholdCostResult:
    threshold: float
    total_cost: float
    n_fp: int
    n_fn: int
    n_tp: int
    n_tn: int
    fp_cost_total: float
    fn_cost_total: float
    f1: float
    precision: float
    recall: float


def compute_cost_at_threshold(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    threshold: float,
    config: CostConfig,
    amounts: Optional[ArrayLike] = None,
) -> ThresholdCostResult:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    fn_costs = config.fn_cost_for_amounts(amounts, len(y_true_arr))
    # Cost only on actual FN rows
    fn_mask = (y_true_arr == 1) & (y_pred == 0)
    fn_cost_total = float(fn_costs[fn_mask].sum())
    fp_cost_total = float(fp * config.cost_fp)
    total = fp_cost_total + fn_cost_total

    prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    return ThresholdCostResult(
        threshold=float(threshold),
        total_cost=total,
        n_fp=fp,
        n_fn=fn,
        n_tp=tp,
        n_tn=tn,
        fp_cost_total=fp_cost_total,
        fn_cost_total=fn_cost_total,
        f1=f1,
        precision=prec,
        recall=rec,
    )


def select_cost_minimizing_threshold(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    config: CostConfig | None = None,
    amounts: Optional[ArrayLike] = None,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Sweep thresholds on validation data; return the cost-optimal threshold
    plus the full cost curve for plotting.
    """
    config = config or CostConfig()
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    curve: List[Dict[str, Any]] = []
    best: Optional[ThresholdCostResult] = None

    for t in thresholds:
        res = compute_cost_at_threshold(y_true, y_prob, float(t), config, amounts=amounts)
        curve.append(asdict(res))
        if best is None or res.total_cost < best.total_cost:
            best = res
        elif best is not None and res.total_cost == best.total_cost and res.f1 > best.f1:
            best = res

    assert best is not None
    return {
        "config": asdict(config),
        "best": asdict(best),
        "curve": curve,
    }


def compare_f1_vs_cost_thresholds(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    f1_threshold: float,
    config: CostConfig | None = None,
    amounts: Optional[ArrayLike] = None,
) -> Dict[str, Any]:
    """Compare F1-optimal vs cost-optimal thresholds on the same probabilities."""
    config = config or CostConfig()
    cost_search = select_cost_minimizing_threshold(y_true, y_prob, config=config, amounts=amounts)
    f1_res = compute_cost_at_threshold(y_true, y_prob, f1_threshold, config, amounts=amounts)
    return {
        "config": asdict(config),
        "f1_threshold_result": asdict(f1_res),
        "cost_threshold_result": cost_search["best"],
        "cost_curve": cost_search["curve"],
        "savings_vs_f1_threshold": float(f1_res.total_cost - cost_search["best"]["total_cost"]),
    }


def save_cost_artifacts(
    comparison: Dict[str, Any],
    prefix: str = "cost_analysis",
) -> Dict[str, str]:
    """Persist cost curve CSV/JSON and a matplotlib cost-vs-threshold plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    curve_df = pd.DataFrame(comparison["cost_curve"])
    csv_path = ARTIFACTS_DIR / f"{prefix}_curve.csv"
    json_path = METRICS_DIR / f"{prefix}.json"
    fig_path = FIGURES_DIR / f"{prefix}_curve.png"

    curve_df.to_csv(csv_path, index=False)
    # Persist without the full curve duplicate in JSON body for size; keep summary + sample
    payload = {
        "config": comparison["config"],
        "f1_threshold_result": comparison["f1_threshold_result"],
        "cost_threshold_result": comparison["cost_threshold_result"],
        "savings_vs_f1_threshold": comparison["savings_vs_f1_threshold"],
        "curve_rows": len(curve_df),
        "curve_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(curve_df["threshold"], curve_df["total_cost"], color="#264653", label="Total cost")
    ax.axvline(
        comparison["cost_threshold_result"]["threshold"],
        color="#E76F51",
        linestyle="--",
        label=f"Cost-opt thr={comparison['cost_threshold_result']['threshold']:.2f}",
    )
    ax.axvline(
        comparison["f1_threshold_result"]["threshold"],
        color="#2A9D8F",
        linestyle=":",
        label=f"F1 thr={comparison['f1_threshold_result']['threshold']:.2f}",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Estimated $ cost")
    ax.set_title("Validation cost vs decision threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    return {"csv": str(csv_path), "json": str(json_path), "figure": str(fig_path)}
