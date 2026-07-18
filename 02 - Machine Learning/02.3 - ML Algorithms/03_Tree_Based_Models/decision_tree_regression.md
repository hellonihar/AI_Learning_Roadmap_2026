# Decision Tree (Regression)

Same structure as classification trees, but leaves predict a **continuous value** (the mean of training samples in that leaf) instead of a class.

## How It Works

Split criteria change from purity measures to **variance reduction**:

- Choose the split that minimizes MSE (or MAE) in child nodes
- Each leaf predicts the **average target value** of training samples in that leaf

```
Leaf prediction = mean(y₁, y₂, ..., yₙ) for samples in that leaf
```

## Key Differences from Classification Trees

| Aspect | Classification | Regression |
|---|---|---|
| Leaf prediction | Majority class | Mean of target |
| Split criterion | Gini / Entropy | MSE / MAE |
| Output | Discrete class | Continuous number |
| Decision boundary | Hard regions | Piecewise constant |

## Characteristics

- Creates a **step-function** approximation of the target (piecewise constant)
- More steps (leaves) = more flexible = higher overfitting risk
- Cannot extrapolate beyond training range (each leaf predicts the training mean)

## Examples

1. **House price prediction**: First split: `sqft > 1500?`. Leaf prices: sqft > 1500 → avg $450K; sqft < 1500 → avg $280K. Further splits refine within these groups.
2. **Delivery time estimation**: `distance > 10km?` → `is_rush_hour?` → `historical_driver_speed`. Each leaf contains similar trips, predicts average delivery time.
3. **Energy consumption forecasting**: `temperature > 25°C?` → `is_weekend?` → `hour > 18?`. Each leaf predicts average kWh for that scenario.
