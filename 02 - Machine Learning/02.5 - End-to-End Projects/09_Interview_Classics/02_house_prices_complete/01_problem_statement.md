# House Prices — Problem Statement

## Business Context
A real estate investment firm wants to accurately estimate home sale prices from property attributes. Accurate valuation improves bidding strategy, renovation ROI analysis, and portfolio risk assessment.

## Problem Type
Regression — predict continuous sale price (log-transformed to handle skew).

## Success Metrics
- **RMSE** < 0.15 (on log-transformed prices) — competitive with top Kaggle entries.
- **R²** ≥ 0.90.
- Demonstrate feature engineering from 79 raw features.

## Constraints
- Many features have missing values (pool, alley, fireplace, etc.) — missingness itself is informative.
- Target is right-skewed — log-transform required.
- Must handle both numeric and categorical features with complex missing patterns.
