# Fraud Detection — Results

## Metrics

| Model | PR AUC | Recall @ 5% FPR | F2 Score |
|-------|--------|------------------|----------|
| Isolation Forest | 0.12 | 0.08 | 0.15 |
| XGBoost (no tuning) | 0.41 | 0.52 | 0.38 |
| XGBoost + threshold tuning | 0.45 | 0.61 | 0.52 |

## What Was Learned

- Frequency encoding of card1 and email domain gave the strongest single-feature lift (+3% PR AUC)
- TransactionAmt alone is a poor fraud indicator; the ratio of amt to card-level average was much more informative
- The optimal threshold was ~0.15, not the default 0.5 — catching fraud requires aggressive flagging
- Model confidence was well-calibrated for high-score predictions but overconfident in the mid-range

## Failure Cases

- **First-party fraud** — transactions where the cardholder themselves files a chargeback (looks identical to legitimate)
- **Slow fraud** — fraudsters who build history over 30+ days before attacking
- **Null-intensive rows** — when >60% identity features are missing, predictions regress to the mean
