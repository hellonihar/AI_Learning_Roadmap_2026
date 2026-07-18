# Lasso Regression

Linear regression with **L1 regularization**. Adds a penalty proportional to the absolute value of coefficient magnitudes.

```
Loss = MSE + λ * Σ|wᵢ|
```

## Key Difference from Ridge

Lasso can shrink coefficients **exactly to zero** — it performs automatic feature selection.

| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| Penalty | Σ(w²) | Σ\|w\| |
| Shrinks to zero? | Never | Yes |
| Feature selection? | No | Yes |
| Best when | Many relevant features | Few relevant features |

## λ Tuning

Higher λ → more coefficients become zero → simpler model.

**Example**: With 100 features:
- λ = 0: all 100 features used (linear regression)
- λ = 0.01: 45 features have non-zero coefficients
- λ = 0.1: 12 features remain
- λ = 1: 3 features remain
- λ = 10: all coefficients zero (predict mean)

## When to Use

- You believe only a subset of features are truly relevant
- Interpretability is important (sparse model = easier to explain)
- Feature selection as part of model training

## Examples

1. **Marketing attribution**: 50 ad channels, but only 5 actually drive sales. Lasso zeroes out the 45 ineffective channels, leaving a clean interpretable model: "only TV, search, email, social, and referral matter."
2. **Medical diagnosis**: 200 lab test results, but only 8 strongly predict a disease. Lasso selects the 8 key tests, reducing future testing costs.
3. **Text classification**: 10,000 TF-IDF features. Lasso selects the 200 most discriminative words, producing a sparse, fast, interpretable classifier.
