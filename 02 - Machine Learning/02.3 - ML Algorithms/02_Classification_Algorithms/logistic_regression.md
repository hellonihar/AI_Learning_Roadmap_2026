# Logistic Regression

Despite the name, it's a **classification** algorithm. Predicts the probability that a sample belongs to a class.

## How It Works

Passes the linear output through a **sigmoid function** to squash it to [0, 1]:

```
p = 1 / (1 + e^-(w₁x₁ + w₂x₂ + ... + b))
```

- If p ≥ 0.5 → predict class 1
- If p < 0.5 → predict class 0

## Odds Ratio Interpretation

Each coefficient tells you how the **log-odds** of the target change per unit of the feature.

`e^w` = odds ratio. A coefficient of 0.5 means a 1-unit increase multiplies the odds of the positive class by e^0.5 ≈ 1.65.

## Multi-Class (Softmax)

For 3+ classes, use **softmax regression** (multinomial logistic regression). Each class gets its own coefficient vector, and softmax converts the scores to probabilities that sum to 1.

## Examples

1. **Spam detection**: Input features = word counts, email metadata. Output = P(spam). Coefficients reveal which words are most spam-indicative: "free" (OR = 3.2), "winner" (OR = 4.1).
2. **Loan default prediction**: Features = income, credit score, debt ratio. Output = P(default). Banks use this for decision-making because coefficients are auditable and explainable to regulators.
3. **Medical diagnosis**: Features = patient age, blood pressure, cholesterol, BMI. Output = P(heart disease). Logistic regression is common in clinical prediction rules because doctors need to understand why a patient was flagged.
