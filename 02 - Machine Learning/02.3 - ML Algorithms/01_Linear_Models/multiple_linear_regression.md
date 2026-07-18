# Multiple Linear Regression

Extension of simple linear regression to **multiple features**. The same core idea — learn weights for each predictor.

```
y = w₁x₁ + w₂x₂ + ... + wₚxₚ + b
```

## When to Use

Any regression problem with 2+ features where the relationship is approximately linear.

## Multicollinearity

When two features are highly correlated, their coefficients become unstable — small changes in data can cause large swings in weights.

**Detection**: Variance Inflation Factor (VIF) > 10 indicates a problem.

**Fixes**: Remove one of the correlated features; use Ridge Regression; combine correlated features into a single feature.

## Feature Scaling

Coefficients are sensitive to feature scale. A feature measured in dollars vs thousands of dollars will have very different coefficient magnitudes. Always scale features for interpretable coefficients.

## Examples

1. **Car price prediction**: `price = 5,000 × horsepower + 200 × top_speed — 3,000 × age_years + 8,000 × luxury_brand — 15,000`. Each coefficient reveals the marginal contribution of each feature.
2. **Student performance**: `final_grade = 0.3 × midterm + 0.2 × homework_avg + 0.4 × final_exam + 5× attendance_rate`. Interpretable grade formula that could be explained to students.
3. **Energy consumption**: `energy_kwh = 1.5 × sqft + 0.8 × occupants — 2.0 × insulation_score + 10 × ac_hours`. Building managers can see exactly what drives energy use.
