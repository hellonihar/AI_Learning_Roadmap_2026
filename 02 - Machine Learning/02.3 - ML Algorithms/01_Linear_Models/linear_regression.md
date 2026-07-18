# Linear Regression

Predict a continuous target using a linear combination of features.

## How It Works

Finds the line (or hyperplane) that minimizes the sum of squared errors between predictions and actual values.

```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

- **w** = weights (coefficients) learned from data
- **b** = bias term (intercept)
- Training = finding w, b that minimize MSE

## Key Assumptions

1. **Linearity**: Relationship between features and target is linear
2. **Independence**: Observations are independent of each other
3. **Homoscedasticity**: Variance of errors is constant across all predictions
4. **Normality of errors**: Errors are normally distributed (needed for inference, not prediction)

In practice, linear regression is robust to mild violations of these assumptions.

## Interpretation

Coefficients have direct meaning: "a 1-unit increase in x₁ is associated with a w₁ change in y, holding all other features constant."

## Examples

1. **House price prediction**: `price = 300 × sqft + 25,000 × bedrooms — 50,000 × age + 100,000`. The coefficient 300 means each additional square foot adds $300 to the predicted price.
2. **Sales forecasting**: `sales = 2.5 × ad_spend + 1.2 × promotion_days + 500`. Every $1K in ad spend yields $2.5K in sales — directly measurable ROI.
3. **Employee performance**: `performance_score = 0.3 × hours_trained + 0.5 × experience_years + 2.0`. Simple, explainable to HR stakeholders.
