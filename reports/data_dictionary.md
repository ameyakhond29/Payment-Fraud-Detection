# Data Dictionary

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
