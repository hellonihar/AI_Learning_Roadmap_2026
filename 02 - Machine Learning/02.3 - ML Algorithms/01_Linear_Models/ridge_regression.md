# Ridge Regression

Linear regression with **L2 regularization**. Adds a penalty proportional to the square of coefficient magnitudes.

```
Loss = MSE + λ * Σ(wᵢ²)
```

## Key Effect

- Shrinks coefficients toward zero (but never exactly zero)
- Reduces variance at the cost of slightly increased bias
- Stabilizes coefficients when multicollinearity exists

## λ (Lambda) Tuning

| λ | Effect |
|---|---|
| λ = 0 | Regular linear regression |
| λ → very small | Mild shrinkage, stable |
| λ → moderate | Significant shrinkage, reduced overfitting |
| λ → very large | All coefficients near zero — underfitting |

**Select λ** via cross-validation. Scikit-learn's `RidgeCV` does this automatically.

## When to Use

- Many features (potentially more than samples)
- High multicollinearity
- Baseline before trying Lasso or ElasticNet

## Examples

1. **Genomics (p >> n)**: 20,000 gene expression features, only 200 patient samples. Ridge stabilizes the model by shrinking most gene coefficients toward zero, keeping only the strongest signals.
2. **Economic forecasting**: GDP, inflation, unemployment, interest rates are all correlated. Ridge handles multicollinearity gracefully — coefficients remain stable even with correlated predictors.
3. **Recommendation with many features**: 500 user behavior features, many correlated. Ridge prevents any single feature from dominating while retaining all features' contributions.
