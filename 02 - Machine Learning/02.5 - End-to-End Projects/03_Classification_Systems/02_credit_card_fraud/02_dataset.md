# Dataset — Credit Card Fraud Detection

## Source
- **Kaggle Credit Card Fraud Detection** (Machine Learning Group - ULB).
- Real transactions from European cardholders, September 2013.

## Size
- 284,807 transactions, 31 columns.
- 492 fraud cases (0.172%) — extremely imbalanced.

## Features
- `V1`–`V28`: PCA-transformed numerical features (original raw data withheld for confidentiality).
- `Amount`: transaction amount (skewed, needs scaling).
- `Time`: seconds elapsed between first transaction and this one.

## Target
- `Class`: 0 (legitimate) / 1 (fraud).

## Known Challenges
- Extreme class imbalance (0.17% fraud) — standard accuracy would exceed 99.8% by predicting all legitimate.
- PCA features are anonymized — no direct business interpretation.
- Temporal leakage possible — `Time` correlates with fraud patterns.
- Duplicate or near-duplicate transactions exist.

## Preprocessing
- StandardScaler on `Amount`.
- Drop `Time` or bin it into hourly features to avoid leakage.
- No PCA inversion possible — features must be used as-is.
