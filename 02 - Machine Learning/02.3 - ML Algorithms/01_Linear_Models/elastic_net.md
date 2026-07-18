# Elastic Net

Combines L1 (Lasso) and L2 (Ridge) penalties. Best of both worlds.

```
Loss = MSE + λ₁ * Σ|wᵢ| + λ₂ * Σ(wᵢ²)
```

Or equivalently (using the common parameterization):

```
Loss = MSE + α * [r * L1 + (1 — r) * L2]
```

## Two Hyperparameters

| Parameter | Controls | Effect |
|---|---|---|
| α (alpha) | Overall regularization strength | Higher = stronger penalty |
| r (l1_ratio) | Balance between L1 and L2 | 0 = Ridge, 1 = Lasso |

## Why Elastic Net Over Pure Lasso?

| Scenario | Lasso | Elastic Net |
|---|---|---|
| Correlated features | Picks one arbitrarily, drops the rest | Keeps all, shrinks together |
| p >> n (many features) | Can only select up to n features | Can select more than n features |
| Few true features | Works well | Works equally well |
| Grouped features (dummy variables) | May split groups | Keeps groups together |

## When to Use

- You have many correlated features
- You want feature selection but with group stability
- You're unsure whether L1 or L2 is more appropriate — let data decide via cross-validation

## Examples

1. **Gene-disease association**: 10,000 gene expressions, many correlated (genes work in pathways). Lasso arbitrarily picks one gene per pathway; Elastic Net keeps the whole pathway, producing more biologically meaningful results.
2. **Credit risk modeling**: 300 features with grouped dummy variables (50 job types, 50 regions). Lasso might select only 3 job types and 2 regions — unstable across samples. Elastic Net keeps more groups coherent.
3. **Demand forecasting**: 200 features including price, promotion, weather, holidays (many interactions). Elastic Net selects relevant groups while handling collinearity between related features like temperature and season.
