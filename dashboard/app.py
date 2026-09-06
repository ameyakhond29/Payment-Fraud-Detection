"""
Streamlit dashboard for ULB Credit Card Fraud Detection.

Includes:
  - KPI metrics + confusion matrix
  - Fraud by hour / amount bin
  - Cost-aware threshold comparison
  - SHAP / feature importance

Run:
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import (
    ARTIFACTS_DIR,
    CLEAN_CSV,
    DATA_PROCESSED,
    FIGURES_DIR,
    METRICS_DIR,
    METRICS_SUMMARY,
)

st.set_page_config(page_title="Payment Fraud Monitor", page_icon="🛡️", layout="wide")


@st.cache_data
def load_metrics() -> dict:
    path = METRICS_SUMMARY if METRICS_SUMMARY.exists() else METRICS_DIR / "metrics_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_clean() -> pd.DataFrame:
    if CLEAN_CSV.exists():
        return pd.read_csv(CLEAN_CSV, parse_dates=["timestamp"])
    return pd.DataFrame()


@st.cache_data
def load_predictions() -> pd.DataFrame:
    path = ARTIFACTS_DIR / "test_predictions.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["timestamp"])
    alt = DATA_PROCESSED / "test_predictions_tableau.csv"
    if alt.exists():
        return pd.read_csv(alt, parse_dates=["timestamp"])
    return pd.DataFrame()


@st.cache_data
def load_cost_curve() -> pd.DataFrame:
    path = ARTIFACTS_DIR / "cost_analysis_curve.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_shap() -> pd.DataFrame:
    path = ARTIFACTS_DIR / "shap_mean_abs.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_importance() -> pd.DataFrame:
    path = ARTIFACTS_DIR / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def main() -> None:
    st.title("Payment Fraud Detection & Transaction Monitoring")
    st.caption(
        "ULB Credit Card Fraud dataset (`creditcard.csv`). "
        "Operational predictions use the **cost-optimal** threshold."
    )

    metrics = load_metrics()
    clean = load_clean()
    preds = load_predictions()
    cost_curve = load_cost_curve()
    shap_df = load_shap()
    imp_df = load_importance()

    if not metrics or clean.empty:
        st.warning("No pipeline outputs found. Run: `python run_pipeline.py`")
        return

    best = metrics.get("best_model", "xgboost")
    # Prefer cost-aware test metrics for KPIs when present
    cost_block = metrics.get("cost_aware", {})
    cost_test = cost_block.get("test", {})
    best_m = cost_test.get("cost_threshold_metrics") or metrics.get("best_model_test_metrics", {})
    ds = metrics.get("dataset", {})
    cfg = metrics.get("cost_config") or cost_block.get("config") or {}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Transactions", f"{ds.get('n_clean_transactions', len(clean)):,}")
    c2.metric("Fraud Rate", f"{ds.get('fraud_rate_clean', clean['is_fraud'].mean()):.3%}")
    c3.metric("Precision", f"{best_m.get('precision', 0):.3f}")
    c4.metric("Recall", f"{best_m.get('recall', 0):.3f}")
    c5.metric("ROC-AUC", f"{best_m.get('roc_auc', 0):.3f}")

    c6, c7, c8, c9 = st.columns(4)
    c6.metric("F1-Score", f"{best_m.get('f1', 0):.3f}")
    c7.metric("PR-AUC", f"{best_m.get('pr_auc', 0):.3f}")
    fp = best_m.get("false_positives", best_m.get("confusion_matrix", {}).get("fp", 0))
    c8.metric("False Positives (test)", f"{fp:,}")
    thr = best_m.get("threshold", cost_test.get("cost_threshold_cost", {}).get("threshold"))
    c9.metric("Cost threshold", f"{thr:.3f}" if thr is not None else "—")

    st.info(
        "Public ULB data has no location/payment-method fields. "
        "Monitoring uses **hour-of-day** and **amount bins**. "
        f"Best ranking model by PR-AUC: **{best}**."
    )

    tab_overview, tab_cost, tab_explain, tab_preds = st.tabs(
        ["Overview", "Cost-aware threshold", "Explainability (SHAP)", "Predictions"]
    )

    with tab_overview:
        left, right = st.columns(2)
        with left:
            st.subheader("Fraud by Hour")
            if "hour" not in clean.columns:
                clean = clean.copy()
                clean["hour"] = ((clean["seconds_from_start"] % 86400) // 3600).astype(int)
            loc = (
                clean.groupby("hour")
                .agg(transactions=("is_fraud", "count"), fraud=("is_fraud", "sum"), fraud_rate=("is_fraud", "mean"))
                .reset_index()
            )
            fig = px.bar(
                loc, x="hour", y="fraud_rate", hover_data=["transactions", "fraud"],
                color="fraud_rate", color_continuous_scale="YlOrRd",
                title="Fraud Rate by Hour of Day",
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Fraud by Amount Bin")
            if "amount_bin" not in clean.columns:
                clean = clean.copy()
                clean["amount_bin"] = pd.cut(
                    clean["amount"].clip(lower=0),
                    bins=[-0.01, 1, 25, 50, 100, 250, 500, 1000, 5000, 1e9],
                    labels=["0-1", "1-25", "25-50", "50-100", "100-250", "250-500", "500-1000", "1000-5000", "5000+"],
                ).astype(str)
            pay = (
                clean.groupby("amount_bin")
                .agg(transactions=("is_fraud", "count"), fraud=("is_fraud", "sum"), fraud_rate=("is_fraud", "mean"))
                .reset_index()
            )
            order = ["0-1", "1-25", "25-50", "50-100", "100-250", "250-500", "500-1000", "1000-5000", "5000+"]
            pay["amount_bin"] = pd.Categorical(pay["amount_bin"], categories=order, ordered=True)
            pay = pay.sort_values("amount_bin")
            fig = px.bar(pay, x="amount_bin", y="fraud_rate", hover_data=["transactions", "fraud"],
                         title="Fraud Rate by Amount Bin")
            fig.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        m1, m2 = st.columns(2)
        with m1:
            st.subheader(f"Confusion Matrix — {best} (cost threshold)")
            cm = best_m.get("confusion_matrix", {})
            z = [[cm.get("tn", 0), cm.get("fp", 0)], [cm.get("fn", 0), cm.get("tp", 0)]]
            fig = go.Figure(
                data=go.Heatmap(
                    z=z,
                    x=["Pred Legit", "Pred Fraud"],
                    y=["Actual Legit", "Actual Fraud"],
                    text=[[str(z[0][0]), str(z[0][1])], [str(z[1][0]), str(z[1][1])]],
                    texttemplate="%{text}",
                    colorscale="Blues",
                    showscale=False,
                )
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with m2:
            st.subheader("All Models — Test Metrics (F1 threshold)")
            rows = []
            for name, m in metrics.get("models", {}).items():
                rows.append(
                    {
                        "Model": name,
                        "Precision": m["precision"],
                        "Recall": m["recall"],
                        "F1": m["f1"],
                        "ROC-AUC": m["roc_auc"],
                        "PR-AUC": m["pr_auc"],
                        "FP": m["false_positives"],
                    }
                )
            st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    with tab_cost:
        st.subheader("Business cost model")
        st.markdown(
            f"""
- **False positive cost:** `${cfg.get('cost_fp', 10):.2f}` per blocked/reviewed legitimate txn  
- **False negative cost:** transaction **amount** (min `${cfg.get('min_fn_cost', 1):.2f}`) when amount is known  
- Threshold is chosen on **validation** to minimize total $ cost, then applied on **test**
"""
        )
        if cost_test:
            f1c = cost_test.get("f1_threshold_cost", {})
            coc = cost_test.get("cost_threshold_cost", {})
            k1, k2, k3 = st.columns(3)
            k1.metric("Test cost @ F1 thr", f"${f1c.get('total_cost', 0):,.2f}")
            k2.metric("Test cost @ cost thr", f"${coc.get('total_cost', 0):,.2f}")
            k3.metric("Test savings", f"${cost_test.get('test_savings_vs_f1', 0):,.2f}")

            cmp = pd.DataFrame(
                [
                    {
                        "Policy": "F1-optimal",
                        "Threshold": f1c.get("threshold"),
                        "FP": f1c.get("n_fp"),
                        "FN": f1c.get("n_fn"),
                        "FP $": f1c.get("fp_cost_total"),
                        "FN $": f1c.get("fn_cost_total"),
                        "Total $": f1c.get("total_cost"),
                    },
                    {
                        "Policy": "Cost-optimal",
                        "Threshold": coc.get("threshold"),
                        "FP": coc.get("n_fp"),
                        "FN": coc.get("n_fn"),
                        "FP $": coc.get("fp_cost_total"),
                        "FN $": coc.get("fn_cost_total"),
                        "Total $": coc.get("total_cost"),
                    },
                ]
            )
            st.dataframe(cmp, use_container_width=True, hide_index=True)

        if not cost_curve.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=cost_curve["threshold"],
                    y=cost_curve["total_cost"],
                    mode="lines",
                    name="Total $ cost",
                    line=dict(color="#264653"),
                )
            )
            if cost_test:
                fig.add_vline(
                    x=cost_test.get("cost_threshold_cost", {}).get("threshold", 0),
                    line_dash="dash",
                    line_color="#E76F51",
                    annotation_text="cost-opt",
                )
                fig.add_vline(
                    x=cost_test.get("f1_threshold_cost", {}).get("threshold", 0),
                    line_dash="dot",
                    line_color="#2A9D8F",
                    annotation_text="F1",
                )
            fig.update_layout(
                title="Validation cost vs threshold",
                xaxis_title="Threshold",
                yaxis_title="Estimated $ cost",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Cost curve artifact missing. Re-run `python run_pipeline.py`.")

        png = FIGURES_DIR / "cost_analysis_curve.png"
        if png.exists():
            st.image(str(png), caption="Saved cost curve figure")

    with tab_explain:
        st.subheader("Feature importance & SHAP")
        e1, e2 = st.columns(2)
        with e1:
            if not shap_df.empty:
                top = shap_df.head(20).iloc[::-1]
                fig = px.bar(
                    top, x="mean_abs_shap", y="feature", orientation="h",
                    title=f"Mean |SHAP| — {best}",
                    color="mean_abs_shap", color_continuous_scale="Teal",
                )
                fig.update_layout(height=520, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("SHAP artifact missing.")
            shap_png = FIGURES_DIR / "shap_summary.png"
            if shap_png.exists():
                st.image(str(shap_png))

        with e2:
            if not imp_df.empty:
                col = "importance" if "importance" in imp_df.columns else imp_df.columns[-1]
                top = imp_df.head(20).iloc[::-1]
                fig = px.bar(
                    top, x=col, y="feature", orientation="h",
                    title=f"Native importance — {best}",
                    color=col, color_continuous_scale="Oranges",
                )
                fig.update_layout(height=520, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Feature importance artifact missing.")
            imp_png = FIGURES_DIR / "feature_importance.png"
            if imp_png.exists():
                st.image(str(imp_png))

        expl = metrics.get("explainability", {})
        if expl:
            st.caption(
                f"Explained {expl.get('n_samples_explained', '—')} test samples. "
                f"SHAP available: {expl.get('artifacts', {}).get('shap_available', False)}"
            )

    with tab_preds:
        st.subheader("High-risk predictions (test set, cost threshold)")
        if not preds.empty:
            cols = [
                c
                for c in [
                    "transaction_id",
                    "timestamp",
                    "amount",
                    "hour",
                    "amount_bin",
                    "is_fraud",
                    "fraud_probability",
                    "predicted_fraud",
                    "predicted_fraud_f1",
                    "false_positive",
                    "false_negative",
                ]
                if c in preds.columns
            ]
            view = preds.sort_values("fraud_probability", ascending=False).head(50)[cols]
            st.dataframe(view, use_container_width=True)

            if "false_positive" in preds.columns:
                fp_df = preds[preds["false_positive"] == 1]
                st.metric("False positives (cost policy)", f"{len(fp_df):,}")
        else:
            st.info("Prediction artifact not found.")

    with st.expander("Pipeline metadata"):
        st.json(
            {
                "best_model": best,
                "dataset": ds,
                "cost_config": cfg,
                "cost_aware_summary": {
                    "test_savings_vs_f1": cost_test.get("test_savings_vs_f1"),
                    "cost_threshold": cost_test.get("cost_threshold_cost", {}).get("threshold"),
                    "f1_threshold": cost_test.get("f1_threshold_cost", {}).get("threshold"),
                },
                "split": metrics.get("split"),
            }
        )


if __name__ == "__main__":
    main()
