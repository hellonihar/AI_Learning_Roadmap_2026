# Regression Losses (MAE, MSE — Intuition)

Both measure how far predictions are from actual values. They differ in **how they penalize errors**.

## Mean Absolute Error (MAE)

Average of absolute differences: `MAE = (1/n) * Σ |actual — predicted|`

- Each error gets the same weight (linear penalty)
- **Robust to outliers** — one bad prediction doesn't dominate
- Consequence: the optimal MAE model predicts the **median**

### Examples
1. **House price prediction**: A \$1M mansion predicted as \$900K → error = \$100K. A \$200K condo predicted as \$100K → error = \$100K. MAE treats both errors equally. If one mansion was a data entry mistake (listed as \$1M but worth \$300K), MAE won't distort the model too much trying to fix that one point.
2. **Weather temperature forecasting**: A prediction of 25°C when actual is 30°C → error = 5°C. Two such errors = total 10°C. The model isn't overly penalized for one extremely wrong prediction.
3. **Delivery time estimation**: Under-promise by 10 min vs over-promise by 10 min → same MAE penalty. MAE doesn't care about direction.

## Mean Squared Error (MSE)

Average of squared differences: `MSE = (1/n) * Σ (actual — predicted)²`

- Larger errors get **quadratically more penalty**
- **Sensitive to outliers** — one terrible prediction can dominate the loss
- Consequence: the optimal MSE model predicts the **mean**

### Examples
1. **Stock price prediction**: Predicted \$100, actual \$101 → squared error = 1. But if one prediction is \$100 vs actual \$200 → squared error = 10,000. The model will sacrifice all other predictions to improve this one outlier. MSE is unstable with extreme values.
2. **Object detection bounding box**: A box that's off by 5 pixels → error 25. Off by 50 pixels → error 2500. MSE will heavily bias the model toward avoiding large misalignments, which is usually what you want.
3. **Sensor calibration**: Most readings are off by 0.1 units, but one faulty sensor reads off by 100. MSE will try very hard to fix that one faulty sensor, possibly distorting the calibration for all other sensors.

## Comparison

| | MAE | MSE |
|---|---|---|
| Penalty for small errors | Small | Small |
| Penalty for large errors | Large | **Very large** (squared) |
| Outlier sensitivity | Low | High |
| Optimal prediction | Median | Mean |
| Unit | Same as target | Target² (harder to interpret) |

## RMSE (Root Mean Squared Error)

`RMSE = √MSE` — brings the error back to the original unit. Most commonly reported regression metric because it's in the same unit as the target.

**Example**: House prices in dollars. MSE might be 2.5 billion ($²). RMSE = $50,000 — "on average, the model is off by $50K."
