# Data Quality Report

Dataset: ULB Credit Card Fraud (`creditcard.csv`)

- **Total rows checked:** 284,807
- **Columns:** 35
- **Rows flagged:** 4,661
- **Rows removed:** 1,081
- **Clean rows retained:** 283,726
- **Rules passed:** 9 / 12

## Rule Results

| Rule | Name | Severity | Violations | Rate | Passed |
|------|------|----------|------------|------|--------|
| DQ01 | missing_critical_fields | critical | 0 | 0.0000% | Yes |
| DQ02 | missing_pca_features | critical | 0 | 0.0000% | Yes |
| DQ03 | duplicate_transaction_ids | critical | 0 | 0.0000% | Yes |
| DQ04 | duplicate_feature_rows | warning | 1,081 | 0.3796% | No |
| DQ05 | negative_or_null_amounts | critical | 0 | 0.0000% | Yes |
| DQ06 | invalid_time_values | critical | 0 | 0.0000% | Yes |
| DQ07 | invalid_timestamps | critical | 0 | 0.0000% | Yes |
| DQ08 | invalid_class_labels | critical | 0 | 0.0000% | Yes |
| DQ09 | nonfinite_pca_values | critical | 0 | 0.0000% | Yes |
| DQ10 | amount_label_null_inconsistency | warning | 0 | 0.0000% | Yes |
| DQ11 | extreme_amount_outliers | info | 284 | 0.0997% | No |
| DQ12 | extreme_pca_outliers | info | 3,558 | 1.2493% | No |

## Rule Descriptions

### DQ01: missing_critical_fields
transaction_id, seconds_from_start, amount, is_fraud, timestamp must be present.

### DQ02: missing_pca_features
All PCA features V1–V28 must be present.

### DQ03: duplicate_transaction_ids
transaction_id must be unique.

### DQ04: duplicate_feature_rows
Exact duplicate rows on Time/Amount/Class/V1–V28 are invalid.

### DQ05: negative_or_null_amounts
amount must be numeric and >= 0 (0 is allowed in this dataset).

### DQ06: invalid_time_values
seconds_from_start must be in [0, 300000].

### DQ07: invalid_timestamps
Derived timestamp must parse to a valid datetime.

### DQ08: invalid_class_labels
is_fraud (Class) must be exactly 0 or 1.

### DQ09: nonfinite_pca_values
V1–V28 must be finite (no NaN/Inf).

### DQ10: amount_label_null_inconsistency
Rows with null amount but a class label are inconsistent.

### DQ11: extreme_amount_outliers
Amounts above the 99.9th percentile are flagged (not auto-removed).

### DQ12: extreme_pca_outliers
Rows with any |Vi| above the 99.9th percentile of |Vi| are flagged (not auto-removed).
