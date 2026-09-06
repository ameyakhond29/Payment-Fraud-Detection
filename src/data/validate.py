"""
Data-quality validation for the ULB credit-card fraud dataset.

12 rules covering missing values, duplicates, invalid amounts/times,
non-finite PCA features, and class-label errors.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.paths import (
    CLEAN_CSV,
    DATA_PROCESSED,
    DATA_QUALITY_REPORT,
    PCA_FEATURES,
    REPORTS_DIR,
)


@dataclass
class RuleResult:
    rule_id: str
    name: str
    description: str
    severity: str
    n_violations: int
    n_checked: int
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def violation_rate(self) -> float:
        if self.n_checked == 0:
            return 0.0
        return self.n_violations / self.n_checked


@dataclass
class DataQualityReport:
    total_rows: int
    total_columns: int
    rules: List[RuleResult]
    rows_flagged: int
    rows_removed: int
    clean_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "rows_flagged": self.rows_flagged,
            "rows_removed": self.rows_removed,
            "clean_rows": self.clean_rows,
            "rules": [asdict(r) for r in self.rules],
            "n_rules": len(self.rules),
            "n_rules_passed": sum(1 for r in self.rules if r.passed),
            "n_rules_failed": sum(1 for r in self.rules if not r.passed),
        }


def _mask_missing_critical(df: pd.DataFrame) -> pd.Series:
    cols = ["transaction_id", "seconds_from_start", "amount", "is_fraud", "timestamp"]
    return df[cols].isna().any(axis=1)


def _mask_missing_pca(df: pd.DataFrame) -> pd.Series:
    return df[PCA_FEATURES].isna().any(axis=1)


def _mask_duplicate_txn_ids(df: pd.DataFrame) -> pd.Series:
    return df["transaction_id"].duplicated(keep="first")


def _mask_duplicate_rows(df: pd.DataFrame) -> pd.Series:
    """Mark extra copies of exact duplicate feature rows (keep first)."""
    subset = ["seconds_from_start", "amount", "is_fraud"] + PCA_FEATURES
    return df.duplicated(subset=subset, keep="first")


def _mask_negative_amounts(df: pd.DataFrame) -> pd.Series:
    amt = pd.to_numeric(df["amount"], errors="coerce")
    return amt.isna() | (amt < 0)


def _mask_invalid_time(df: pd.DataFrame) -> pd.Series:
    t = pd.to_numeric(df["seconds_from_start"], errors="coerce")
    # ULB spans ~2 days (~172k seconds); allow small buffer
    return t.isna() | (t < 0) | (t > 300_000)


def _mask_invalid_timestamps(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    return ts.isna()


def _mask_invalid_labels(df: pd.DataFrame) -> pd.Series:
    lab = pd.to_numeric(df["is_fraud"], errors="coerce")
    return lab.isna() | ~lab.isin([0, 1])


def _mask_nonfinite_pca(df: pd.DataFrame) -> pd.Series:
    vals = df[PCA_FEATURES].to_numpy(dtype=float, copy=False)
    return pd.Series(~np.isfinite(vals).all(axis=1), index=df.index)


def _mask_amount_class_inconsistency(df: pd.DataFrame) -> pd.Series:
    """Flag impossible combos: NaN amount with a valid label (caught elsewhere too)."""
    amt = pd.to_numeric(df["amount"], errors="coerce")
    lab = pd.to_numeric(df["is_fraud"], errors="coerce")
    return amt.isna() & lab.isin([0, 1])


def _mask_extreme_amount_outliers(df: pd.DataFrame) -> pd.Series:
    amt = pd.to_numeric(df["amount"], errors="coerce")
    positive = amt[amt >= 0]
    if positive.empty:
        return pd.Series(False, index=df.index)
    threshold = positive.quantile(0.999)
    return amt > threshold


def _mask_extreme_pca_outliers(df: pd.DataFrame) -> pd.Series:
    """Flag rows where any |Vi| exceeds 99.9th pct of |Vi| (info only)."""
    mask = pd.Series(False, index=df.index)
    for col in PCA_FEATURES:
        abs_v = df[col].abs()
        thr = abs_v.quantile(0.999)
        mask |= abs_v > thr
    return mask


RULE_SPECS: List[Dict[str, Any]] = [
    {
        "rule_id": "DQ01",
        "name": "missing_critical_fields",
        "description": "transaction_id, seconds_from_start, amount, is_fraud, timestamp must be present.",
        "severity": "critical",
        "mask_fn": _mask_missing_critical,
        "remove": True,
    },
    {
        "rule_id": "DQ02",
        "name": "missing_pca_features",
        "description": "All PCA features V1–V28 must be present.",
        "severity": "critical",
        "mask_fn": _mask_missing_pca,
        "remove": True,
    },
    {
        "rule_id": "DQ03",
        "name": "duplicate_transaction_ids",
        "description": "transaction_id must be unique.",
        "severity": "critical",
        "mask_fn": _mask_duplicate_txn_ids,
        "remove": True,
    },
    {
        "rule_id": "DQ04",
        "name": "duplicate_feature_rows",
        "description": "Exact duplicate rows on Time/Amount/Class/V1–V28 are invalid.",
        "severity": "warning",
        "mask_fn": _mask_duplicate_rows,
        "remove": True,
    },
    {
        "rule_id": "DQ05",
        "name": "negative_or_null_amounts",
        "description": "amount must be numeric and >= 0 (0 is allowed in this dataset).",
        "severity": "critical",
        "mask_fn": _mask_negative_amounts,
        "remove": True,
    },
    {
        "rule_id": "DQ06",
        "name": "invalid_time_values",
        "description": "seconds_from_start must be in [0, 300000].",
        "severity": "critical",
        "mask_fn": _mask_invalid_time,
        "remove": True,
    },
    {
        "rule_id": "DQ07",
        "name": "invalid_timestamps",
        "description": "Derived timestamp must parse to a valid datetime.",
        "severity": "critical",
        "mask_fn": _mask_invalid_timestamps,
        "remove": True,
    },
    {
        "rule_id": "DQ08",
        "name": "invalid_class_labels",
        "description": "is_fraud (Class) must be exactly 0 or 1.",
        "severity": "critical",
        "mask_fn": _mask_invalid_labels,
        "remove": True,
    },
    {
        "rule_id": "DQ09",
        "name": "nonfinite_pca_values",
        "description": "V1–V28 must be finite (no NaN/Inf).",
        "severity": "critical",
        "mask_fn": _mask_nonfinite_pca,
        "remove": True,
    },
    {
        "rule_id": "DQ10",
        "name": "amount_label_null_inconsistency",
        "description": "Rows with null amount but a class label are inconsistent.",
        "severity": "warning",
        "mask_fn": _mask_amount_class_inconsistency,
        "remove": True,
    },
    {
        "rule_id": "DQ11",
        "name": "extreme_amount_outliers",
        "description": "Amounts above the 99.9th percentile are flagged (not auto-removed).",
        "severity": "info",
        "mask_fn": _mask_extreme_amount_outliers,
        "remove": False,
    },
    {
        "rule_id": "DQ12",
        "name": "extreme_pca_outliers",
        "description": "Rows with any |Vi| above the 99.9th percentile of |Vi| are flagged (not auto-removed).",
        "severity": "info",
        "mask_fn": _mask_extreme_pca_outliers,
        "remove": False,
    },
]


def run_validation(df: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport, pd.DataFrame]:
    n = len(df)
    remove_mask = pd.Series(False, index=df.index)
    flagged_mask = pd.Series(False, index=df.index)
    results: List[RuleResult] = []
    violation_cols: Dict[str, pd.Series] = {}

    for spec in RULE_SPECS:
        mask_fn: Callable[[pd.DataFrame], pd.Series] = spec["mask_fn"]
        mask = mask_fn(df).fillna(False)
        violation_cols[spec["rule_id"]] = mask
        flagged_mask |= mask
        if spec["remove"]:
            remove_mask |= mask
        results.append(
            RuleResult(
                rule_id=spec["rule_id"],
                name=spec["name"],
                description=spec["description"],
                severity=spec["severity"],
                n_violations=int(mask.sum()),
                n_checked=n,
                passed=bool(mask.sum() == 0),
                details={"remove_on_fail": spec["remove"]},
            )
        )

    clean_df = df.loc[~remove_mask].copy()
    # After removing duplicate groups, keep first of any remaining id collisions
    clean_df = clean_df.drop_duplicates(subset=["transaction_id"], keep="first")
    # Also drop exact feature duplicates keeping first
    subset = ["seconds_from_start", "amount", "is_fraud"] + PCA_FEATURES
    clean_df = clean_df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)

    clean_df["amount"] = pd.to_numeric(clean_df["amount"], errors="coerce")
    clean_df["seconds_from_start"] = pd.to_numeric(clean_df["seconds_from_start"], errors="coerce")
    clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], errors="coerce")
    clean_df["is_fraud"] = pd.to_numeric(clean_df["is_fraud"], errors="coerce").astype(int)
    clean_df["hour"] = ((clean_df["seconds_from_start"] % 86400) // 3600).astype(int)

    # Drop original ULB column names that collide with standardized ones in SQLite
    drop_orig = [c for c in ("Time", "Amount", "Class") if c in clean_df.columns]
    if drop_orig:
        clean_df = clean_df.drop(columns=drop_orig)

    violation_log = pd.DataFrame(violation_cols)
    violation_log.insert(0, "transaction_id", df["transaction_id"].values)

    report = DataQualityReport(
        total_rows=n,
        total_columns=df.shape[1],
        rules=results,
        rows_flagged=int(flagged_mask.sum()),
        rows_removed=int(remove_mask.sum()),
        clean_rows=len(clean_df),
    )
    return clean_df, report, violation_log


def write_quality_report(report: DataQualityReport, path: Optional[Path] = None) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = path or DATA_QUALITY_REPORT
    d = report.to_dict()
    lines = [
        "# Data Quality Report",
        "",
        "Dataset: ULB Credit Card Fraud (`creditcard.csv`)",
        "",
        f"- **Total rows checked:** {d['total_rows']:,}",
        f"- **Columns:** {d['total_columns']}",
        f"- **Rows flagged:** {d['rows_flagged']:,}",
        f"- **Rows removed:** {d['rows_removed']:,}",
        f"- **Clean rows retained:** {d['clean_rows']:,}",
        f"- **Rules passed:** {d['n_rules_passed']} / {d['n_rules']}",
        "",
        "## Rule Results",
        "",
        "| Rule | Name | Severity | Violations | Rate | Passed |",
        "|------|------|----------|------------|------|--------|",
    ]
    for r in report.rules:
        lines.append(
            f"| {r.rule_id} | {r.name} | {r.severity} | {r.n_violations:,} | "
            f"{r.violation_rate:.4%} | {'Yes' if r.passed else 'No'} |"
        )
    lines.extend(["", "## Rule Descriptions", ""])
    for r in report.rules:
        lines.append(f"### {r.rule_id}: {r.name}")
        lines.append(r.description)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    json_path = REPORTS_DIR / "metrics" / "data_quality_report.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return path


def save_clean_data(df: pd.DataFrame, path: Path = CLEAN_CSV) -> Path:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_data_dictionary(path: Optional[Path] = None) -> Path:
    from src.utils.paths import DATA_DICTIONARY

    path = path or DATA_DICTIONARY
    content = """# Data Dictionary

## Dataset Source

**ULB Machine Learning Group — Credit Card Fraud Detection**

- Public dataset: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Local file: `creditcard.csv` (project root)
- Transactions by European cardholders over two days
- Features `V1`–`V28` are PCA-transformed for confidentiality
- Only `Time` and `Amount` are provided in original form besides the label

Raw standardized data: `data/raw/`. Cleaned / features: `data/processed/`.

## Schema Gaps vs Ideal Payment Monitoring Fields

| Ideal field | Available in ULB? | How handled |
|-------------|-------------------|-------------|
| Transaction amount | Yes (`Amount`) | Used as `amount` |
| Timestamp | Partial (`Time` = seconds since first txn) | Converted to `timestamp` |
| Location | No | Not fabricated; dashboard uses amount/hour instead |
| Device type | No | Not fabricated |
| Payment method | No | Not fabricated (all are card payments) |
| Customer ID | No | Sequence/velocity features used instead |
| Merchant ID | No | Not available |
| Transaction frequency | Derivable | Rolling/prior features from ordered `Time` |
| Historical behavior | Derivable | Expanding amount stats (past-only) |
| Fraud label | Yes (`Class`) | Mapped to `is_fraud` |

## Standardized Columns

| Column | Type | Description |
|--------|------|-------------|
| transaction_id | string | Generated unique id `T########` |
| Time / seconds_from_start | float | Seconds since first transaction in the dataset |
| timestamp | datetime | `2013-09-01` + Time (analysis convenience) |
| Amount / amount | float | Transaction amount (EUR) |
| amount_bin | string | Binned amount for analytics |
| hour | int | Hour-of-day derived from Time |
| V1–V28 | float | Anonymized PCA features |
| Class / is_fraud | int | 1 = fraud, 0 = legitimate |
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
