# Results — Customer Churn Prediction

## Model Comparison (threshold = 0.4)

| Model | Recall | Precision | F1 | AUC-ROC | Lift@10% |
|-------|--------|-----------|-----|---------|----------|
| XGBoost | **0.86** | 0.61 | **0.71** | **0.843** | **3.4** |
| Random Forest | 0.83 | 0.63 | 0.72 | 0.831 | 3.2 |
| Logistic Regression | 0.74 | 0.57 | 0.64 | 0.792 | 2.5 |

## Confusion Matrix (XGBoost, threshold = 0.4)

```
              Predicted 0    Predicted 1
Actual 0        890           220       ← FP = 220
Actual 1         55           343       ← TP = 343, FN = 55
```

- Missed 55 churners (recall = 0.86) — acceptable for retention outreach.
- 220 false positives — retention team has capacity for ~300 calls/day; this is within budget.

## What Was Learned
- **Winner: XGBoost** — highest recall and lift at top decile.
- **Top features**: `contract_Month-to-month`, `tenure`, `service_count`, `support_tickets_90d`.
- Feature engineering added +8 points recall over raw features alone.

## Failure Cases
- Customers who churn due to pricing (no signal in usage features).
- Long-tenure customers with sudden life change (job loss, relocation) — no feature captures this.
- Customers who churn right after a competitor marketing promotion — external event not in model.

## Next Steps
- Incorporate external data: competitor pricing, regional economic indicators.
- Build a separate early-warning model using only first-30-day features.
- A/B test retention offers driven by model vs. random selection.
