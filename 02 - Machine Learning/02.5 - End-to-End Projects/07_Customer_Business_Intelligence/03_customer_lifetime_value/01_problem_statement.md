# Customer Lifetime Value — Problem Statement

## Business Context
A subscription-based SaaS company needs to segment customers by predicted lifetime value to optimize acquisition spend (CAC). Currently using a single "high/medium/low" manual classification. Need a probabilistic model to predict both purchase frequency and monetary value per customer.

## Problem Type
- **Primary**: Probabilistic modeling (BG/NBD for frequency, Gamma-Gamma for monetary)
- **Secondary**: Regression (baseline comparison for monetary value)

## Success Metrics
- **MAE** on predicted vs. actual 3-month future spend
- Calibration: lift curves (predicted LTV deciles vs. actual mean LTV)
- Business decision improvement: ROI of targeting top-decile customers
- Model convergence: Pareto/NBD log-likelihood stability
