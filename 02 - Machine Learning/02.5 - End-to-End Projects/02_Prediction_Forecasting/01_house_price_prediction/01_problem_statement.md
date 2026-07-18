# Problem Statement — House Price Prediction

## Business Context
A real estate platform wants to provide instant, accurate home valuations to buyers and sellers. Manual appraisals take 3–5 days; an ML model can give estimates in seconds, improving user engagement and trust.

## Problem Type
Supervised regression. Predict the **sale price** (continuous) of residential homes given 79+ structural, spatial, and categorical features.

## Why This Matters
- Buyers: budget planning, fair-offer guidance
- Sellers: optimal listing price, time-on-market reduction
- Platform: increased listing accuracy → higher retention

## Success Metrics
| Metric | Target | Notes |
|--------|--------|-------|
| **RMSE** | ≤ $25,000 | Dollars; penalises large errors |
| **R²** | ≥ 0.90 | Variance explained |
| **MAE** | ≤ $18,000 | Interpretable average error |

## Constraints
- Model must infer on a single row in < 100 ms (API use-case)
- Interpretability desired for regulatory compliance (fair lending)
- Should generalise to unseen neighbourhoods without retraining
