# Results — Credit Risk

## Metrics

| Metric | Score |
|--------|-------|
| AUC-ROC | 0.82 |
| KS Statistic | 0.44 |
| Accuracy | 0.76 |
| Precision (bad) | 0.68 |
| Recall (bad) | 0.61 |

## Scorecard Example

| Feature | Level | Points |
|---------|-------|--------|
| Age | < 25 | -15 |
| Age | 25–40 | +5 |
| Age | 40–55 | +10 |
| Age | > 55 | +15 |
| Credit amount | < 2000 | +20 |
| Credit amount | 2000–5000 | +5 |
| Credit amount | > 5000 | -10 |
| Checking account | little | -20 |
| Checking account | moderate | +5 |
| Checking account | rich | +15 |
| Duration | < 12 months | +10 |
| Duration | 12–36 months | 0 |
| Duration | > 36 months | -15 |

**Base score**: 600. Final score = 600 + sum of points.

## What Was Learned

- **WoE binning** makes the model fully interpretable — each bin's contribution is clear.
- **Logistic regression on WoE features** is equivalent to a scorecard and is regulator-friendly.
- **SHAP** on WoE features shows that checking account status and loan amount dominate.
- The scorecard approach is preferred by banks because it's transparent and auditable.

## Failure Cases

- Model performs poorly on applicants with "no account" (checking = 0) — these are treated as high risk but some are creditworthy.
- Small dataset (1,000 rows) limits granularity of WoE bins.
- Some WoE bins have few samples — need to merge rare categories.
