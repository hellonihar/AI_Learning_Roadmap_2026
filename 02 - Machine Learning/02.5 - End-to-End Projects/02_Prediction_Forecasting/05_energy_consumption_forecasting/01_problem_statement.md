# Problem Statement — Energy Consumption Forecasting

## Business Context
A utility company wants to forecast hourly household energy consumption to balance grid load, schedule peaker plants, and offer time-of-use pricing. Better forecasts reduce reliance on expensive spot-market energy and prevent brownouts.

## Problem Type
Time-series regression (multivariate). Predict **hourly kWh consumption** for individual households using past consumption + weather features.

## Success Metrics
| Metric | Target | Notes |
|--------|--------|-------|
| **RMSE** | ≤ 0.35 kWh | Per hour per household |
| **MAE** | ≤ 0.25 kWh | Average absolute error |
| **R²** | ≥ 0.85 | Baseline (persist last hour): ~0.65 |

## Forecasting Horizon
- **Short-term:** next 24 hours (operational grid management)
- **Medium-term:** next 7 days (maintenance scheduling)

## Constraints
- Model must run hourly, updating with latest meter reading
- Must generalise across diverse household profiles (apartment vs. house, different climates)
- Interpretability needed for regulator audit (why was load predicted high?)
