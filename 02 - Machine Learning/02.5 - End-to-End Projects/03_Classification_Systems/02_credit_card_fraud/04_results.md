# Results — Credit Card Fraud Detection

## Model Comparison

| Model | PR-AUC | Precision | Recall | F2 |
|-------|--------|-----------|--------|-----|
| LightGBM (SMOTE + threshold) | **0.765** | 0.52 | 0.86 | 0.76 |
| Random Forest (SMOTE) | 0.721 | 0.48 | 0.83 | 0.71 |
| Logistic Regression (SMOTE) | 0.543 | 0.31 | 0.79 | 0.58 |
| XGBoost (scale_pos_weight) | 0.748 | 0.50 | **0.87** | 0.75 |

## Confusion Matrix Analysis (LightGBM, threshold = 0.38)

```
              Predicted 0    Predicted 1
Actual 0       85,068         269         ← FP = 269
Actual 1           21         112         ← TP = 112, FN = 21
```

- 112 of 133 fraud cases caught (84.2% recall).
- 269 false positives — manageable for a human review team.
- Missed fraud (21 cases) were low-amount transactions with normal spending patterns.

## What Was Learned
- **Winner: LightGBM** — best PR-AUC with fastest training time.
- **SMOTE + scale_pos_weight** outperformed either technique alone.
- **Threshold post-tuning** improved recall by 12 points over default 0.5.

## Failure Cases
- Fraud from stolen cards used for small test transactions ($0.50–$2.00) before a large purchase.
- Transactions where fraud pattern exactly mimics previous legitimate behavior.
- New merchant categories never seen in training data.

## Next Steps
- Build a graph-based model connecting cards, merchants, and IP addresses.
- Add velocity features (transactions per hour per card).
