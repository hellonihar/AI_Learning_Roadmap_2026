# Prediction & Forecasting

End-to-end projects focused on predicting future outcomes from historical data. Covers regression (house prices, energy), classification (stock direction), and time-series forecasting (sales, demand).

## Projects

| # | Project | Type | Key Libraries | Metric |
|---|---------|------|---------------|--------|
| 1 | **House Price Prediction** | Regression (tabular) | sklearn, XGBoost | RMSE, R² |
| 2 | **Stock Price Movement** | Binary Classification | sklearn, yfinance | Precision, F1 |
| 3 | **Sales Forecasting** | Time-Series | statsmodels, sklearn | MAE, WMAPE |
| 4 | **Demand Forecasting** | Hierarchical TS | statsmodels, xgboost | MAE, RMSE |
| 5 | **Energy Consumption** | Time-Series Regression | sklearn, pandas | RMSE |

## Common Pitfalls Covered
- Temporal leakage from improper train/test splits
- Look-ahead bias in feature engineering
- Seasonality and trend decomposition
- Hierarchical reconciliation
- Retraining and drift monitoring strategies

## Structure
Each project contains:
- `01_problem_statement.md` — business context, problem type, success metrics
- `02_dataset.md` — source, schema, known challenges
- `03_implementation.md` — walkthrough with code snippets
- `04_results.md` — benchmark scores, model selection rationale
- `05_deployment_notes.md` — serving, retraining, monitoring
