# Results — Loan Default Prediction

## Model Comparison

| Model | AUC-ROC | Recall (Default) | Precision (Default) | Interpretability |
|-------|---------|------------------|---------------------|-----------------|
| Logistic Regression | **0.756** | 0.62 | 0.48 | SHAP (exact) |
| Random Forest | 0.773 | 0.67 | 0.52 | SHAP (approximate) |
| XGBoost | **0.781** | 0.69 | 0.53 | SHAP (approximate) |
| Logistic (no regularization) | 0.741 | 0.60 | 0.46 | SHAP (exact) |

## Confusion Matrix (Logistic Regression, default threshold 0.5)

```
              Predicted 0    Predicted 1
Actual 0      132,000        18,000       ← FP = 18k
Actual 1       18,500        31,500       ← TP = 31.5k, FN = 18.5k
```

- AUC-ROC of 0.756 is acceptable for a first-cut credit model.
- Regularized LR lags behind tree models by ~2.5 AUC points.

## What Was Learned
- **Logistic Regression chosen for production** — regulatory requirement for interpretability outweighed the 2.5-point AUC gap.
- **Top SHAP features**: `int_rate` (higher → riskier), `dti`, `loan_inc_ratio`.
- **Grade A loans nearly zero defaults** — model may over-rely on LendingClub's proprietary grade.

## Failure Cases
- Self-employed borrowers with high income but volatile cash flow get misclassified.
- Short credit history (< 2 years) with perfect payment record — model penalizes them unfairly.
- Loans originated just before economic shocks (COVID March 2020) — distribution shift.

## Next Steps
- Implement rejection inference to correct selection bias.
- Add macroeconomic features (unemployment rate, Fed funds rate) for temporal robustness.
