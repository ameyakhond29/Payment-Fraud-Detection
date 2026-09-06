# Feature Dictionary

Total model features: **48** (20 engineered + aggregates, including V1–V28).

Leakage controls: expanding/rolling amount statistics use `shift(1)` so the
current row is excluded. `is_fraud` / `Class` are never model inputs.

Note: ULB data has no customer/merchant IDs, so behavioral features are
computed on the globally time-ordered transaction stream.

| Feature | Description | Category | Leakage control |
|---------|-------------|----------|-----------------|
| `amount` | Transaction amount (EUR) | Transaction | Raw public field |
| `log_amount` | log(1 + amount) | Transaction | No label used |
| `amount_sqrt` | sqrt(amount) | Transaction | No label used |
| `hour` | Hour of day from Time | Temporal | Derived from Time |
| `hour_sin / hour_cos` | Cyclical hour encoding | Temporal | No label used |
| `day_index` | Day index from Time // 86400 | Temporal | No label used |
| `seconds_from_start` | ULB Time field | Temporal | Raw public field |
| `seconds_since_prev` | Gap vs previous transaction | Velocity | diff / past-only |
| `amount_delta_prev` | Amount change vs previous txn | Velocity | diff / past-only |
| `expanding_amount_mean` | Expanding mean of prior amounts | Behavior | shift+expanding |
| `expanding_amount_std` | Expanding std of prior amounts | Behavior | shift+expanding |
| `amount_zscore_expanding` | Amount z-score vs prior stream | Behavior | past stats only |
| `rolling_amount_mean_100` | Rolling mean of prior 100 amounts | Behavior | shift+rolling |
| `rolling_amount_std_100` | Rolling std of prior 100 amounts | Behavior | shift+rolling |
| `rolling_txn_rate_100` | Approx rate from prior inter-arrival | Velocity | past gaps only |
| `pca_l2_norm` | L2 norm of V1–V28 | PCA aggregate | No label used |
| `pca_abs_max / mean / std` | Abs stats over PCA vector | PCA aggregate | No label used |
| `V1–V28` | Anonymized PCA components from ULB | PCA | Public features; not labels |