# Fraud Detection — Dataset

## Source
[Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) — real-world transaction data with identity and transaction tables.

## Size & Shape
- **Train**: 590K transactions, 394 features (after merging TransactionDT, TransactionAmt, ProductCD, card1-6, addr1-2, dist1-2, P_emaildomain, R_emaildomain, plus 300+ Vesta-engineered V* features)
- **Test**: 506K transactions (public/private split)
- **Target**: `isFraud` — binary, ~3.5% fraud rate

## Challenges
- **Extreme imbalance** — fraud rate varies by day of week / hour
- **Missing data** — ~40% of identity features are null (only populated for certain transaction types)
- **Time-based leakage** — TransactionDT is not chronological across train/test split; must use time-based folds
- **High cardinality** — card1-6, email domains have thousands of unique values
