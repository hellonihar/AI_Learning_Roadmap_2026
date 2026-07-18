# Results — House Price Prediction

## Benchmark Scores (Public LB)

| Model | RMSE (raw $) | R² | MAE |
|-------|-------------|-----|-----|
| Linear Regression (raw) | 42,000 | 0.82 | 28,000 |
| Ridge (log-target) | 31,000 | 0.89 | 20,000 |
| XGBoost (tuned) | **24,500** | **0.93** | **16,200** |
| Stacked Ensemble | 23,800 | 0.94 | 15,900 |

## Winning Model — XGBoost

- Captures non-linear interactions (e.g., OverallQual x Neighborhood)
- Handles missing values natively; no need for heavy imputation
- Feature importance highlights: `OverallQual`, `GrLivArea`, `TotalBsmtSF`, `GarageCars`

## What Was Learned

1. **Log-transform the target** — without it, models over-penalise expensive homes and RMSE doubles
2. **Ordinal > one-hot** — quality grades (Ex → Po) need monotonic encoding; one-hot loses rank information
3. **Neighborhood is high-cardinality** — 25+ categories; rare groups need frequency encoding or shrinkage (Ridge helps)
4. **Garage/Bsmt finish matters** — not just size, but whether it's finished/unfinished adds signal

## Failure Modes
- Homes with > 4,000 sqft are systematically under-predicted (too few samples)
- New construction (`YearBuilt` = 2020+ with low `YrSold`) has high error — little comparable data
