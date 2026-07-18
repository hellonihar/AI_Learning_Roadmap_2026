# Regression Metrics (Intuition)

## MAE — Mean Absolute Error

Average absolute difference: "On average, how many dollars/units off are we?"

**Example**: Model predicts house prices. Actual = [200K, 300K, 500K]. Predicted = [210K, 280K, 480K]. Errors = [10K, 20K, 20K]. MAE = 16.7K. "On average, predictions are off by ~$17K."

## MSE — Mean Squared Error

Average squared difference. Penalizes large errors more heavily. **Not in the original unit.**

**Example**: Same housing predictions. Squared errors = [100M, 400M, 400M]. MSE = 300M ($²). Hard to interpret directly.

## RMSE — Root Mean Squared Error

√MSE. **Back in the original unit.** The most commonly reported regression metric.

**Example**: Same housing predictions. RMSE = √300M ≈ $17.3K. "On average, the typical error magnitude is ~$17K." (slightly higher than MAE because large errors are amplified)

## R² — Coefficient of Determination

"Fraction of variance explained by the model." Ranges from -∞ to 1.

- **1**: Perfect predictions
- **0**: Model is as good as predicting the mean
- **Negative**: Model is worse than predicting the mean

**Example**: R² = 0.85 → "The model explains 85% of the variance in house prices. 15% is unexplained (noise, missing features)."

## When to Use Which

| Metric | Best When |
|---|---|
| **MAE** | You care about average error magnitude, outliers are data errors |
| **MSE / RMSE** | Large errors are disproportionately bad (safety-critical) |
| **R²** | You want to explain how much variance the model captures |

## Examples

1. **Demand forecasting for retail**: MAE = 50 units/day. "On average, we miss demand by 50 units." RMSE might be 80 if occasionally a promotion causes a 500-unit miss. The difference between MAE and RMSE tells you about outlier sensitivity.
2. **Autonomous driving braking distance**: RMSE is preferred — a 10-meter error is slightly bad, a 50-meter error is a crash. Squared penalty correctly prioritizes avoiding large errors.
3. **Energy consumption prediction**: R² = 0.92 → "The model explains 92% of consumption variance using temperature, time-of-day, and day-of-week."
