# Medical Cost Prediction — Problem Statement

## Business Context
An insurance provider needs to estimate annual medical costs for individual beneficiaries. Accurate predictions improve premium pricing, reserve allocation, and flagging of high-cost members for care management programs.

## Problem Type
Regression — predict continuous annual medical charges in USD.

## Success Metrics
- **MAE** < $3,000 — average error acceptable for pricing.
- **RMSE** < $8,000 — penalize large underestimates that cause reserve shortfalls.
- **R²** ≥ 0.75 — explain variance in charges.

## Constraints
- Target is right-skewed (most people low cost, few very high).
- Must handle categorical features (region, sex, smoker status).
- Predictions used for pricing — underestimation is worse than overestimation.
