# Polynomial Regression

Extends linear regression by adding polynomial terms of features: `x²`, `x³`, etc.

```
y = w₁x + w₂x² + w₃x³ + b
```

Despite the name, it's still **linear in the coefficients** — the non-linearity comes from the feature transformation, not the model itself.

## Degree Selection

| Degree | Fit | Risk |
|---|---|---|
| 1 (linear) | May underfit simple curves | High bias |
| 2 (quadratic) | Captures one curve (U-shape) | Moderate |
| 3 (cubic) | Captures S-curves | Moderate-high |
| 4+ | Very flexible | **High overfitting risk** |

**Rule**: Start with degree=2. Only increase if validation performance improves. Degree > 5 is rarely useful.

## When to Use

- Clear non-linear relationship (e.g., temperature vs ice cream sales — peaks at moderate temps)
- Small number of features (features explode: 1 feature at degree 5 = 5 terms; 10 features at degree 5 = 2002 terms)
- Interpretability is secondary to fit quality

## Examples

1. **Ice cream sales vs temperature**: Sales increase up to 30°C, then decrease (too hot for outdoor consumption). Linear regression misses this entirely. Polynomial (degree 2) captures the peak naturally.
2. **Drug dosage vs effect**: Low doses have little effect, medium doses have strong effect, high doses plateau. Polynomial (degree 3) captures the S-curve shape.
3. **Car acceleration vs fuel efficiency**: Efficiency drops non-linearly with acceleration. A quadratic term captures the diminishing returns.
