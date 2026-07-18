# Support Vector Regressor (SVR)

Extends SVM to **regression** problems. Instead of maximizing a margin between classes, SVR fits as many data points as possible **within an ε (epsilon) tube** around the regression line.

## How It Works

Unlike linear regression (which minimizes error on all points), SVR:
- Ignores errors smaller than ε (inside the tube)
- Penalizes errors larger than ε (outside the tube)
- Maximizes the margin by making the tube as flat as possible

```
Linear Regression:     SVR:
Minimize ALL errors    Ignore errors < ε (no penalty)
                       Penalize errors > ε
```

## Key Hyperparameters

| Parameter | Effect |
|---|---|
| **ε (epsilon)** | Tube width. Large ε = wider tube (more tolerance, simpler model). Small ε = narrow tube (less tolerance, risk of overfitting). |
| **C** | Penalty for points outside the tube. High C = hard margin. Low C = soft margin. |
| **γ (gamma)** | Kernel coefficient (for RBF). Same intuition as SVC. |

## When to Use SVR

- Small datasets (<5K samples)
- Non-linear relationships
- When you can tolerate higher inference time (SVR is slower than tree-based models)
- When you need a flexible regression model without neural networks

## Examples

1. **Stock price forecasting (short-term)**: 2K days of historical data, features = price + volume + technical indicators. SVR with RBF kernel captures non-linear patterns that linear regression misses.
2. **Energy consumption prediction**: 3K buildings, features = sqft, occupants, HVAC type, weather. SVR handles the non-linear relationship between temperature and energy use better than linear models.
3. **Drug response prediction**: 500 patients, 100 genetic markers. SVR predicts drug effectiveness from genetic features. Small dataset, high dimensions — SVM-family algorithms excel in this regime.
