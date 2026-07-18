# Results — House Prices

## Metrics (5-fold CV on log-transformed target)

| Model | CV RMSE (log) | CV R² | Kaggle Public LB |
|-------|---------------|-------|------------------|
| Elastic Net | 0.142 | 0.872 | 0.138 |
| Gradient Boosting | 0.128 | 0.896 | 0.124 |
| **Stacking (ENet + GBM → Ridge)** | **0.121** | **0.907** | **0.118** |

## Feature Importance (GBM top 10)

1. OverallQual — overall material and finish quality
2. GrLivArea — above-grade living area
3. OverallGrade (interaction) — quality × condition
4. TotalBsmtSF — total basement area
5. GarageCars — garage size
6. YearBuilt — newer homes sell for more
7. TotalSF — total square footage
8. Neighborhood_Crawford
9. GarageArea
10. FullBath

## What Was Learned

- **Missing-as-informative** is critical: NaN in PoolQC, Alley, Fence means "does not exist."
- **Skew correction** on numeric features improves Elastic Net by 8%.
- **Interaction features** (OverallGrade, TotalSF) capture non-linear relationships that base features miss.
- Stacking outperforms individual models by combining linear + non-linear strengths.

## Failure Cases

- Model overestimates for properties with extreme lot sizes (> 100k sqft).
- Underestimates for newly built luxury homes (few training examples).
- Some categorical levels with < 5 samples cause unstable Elastic Net coefficients.
