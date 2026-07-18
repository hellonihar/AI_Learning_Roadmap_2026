# Results — Customer Churn

## Metrics (5-fold CV)

| Model | CV AUC-ROC | Precision@20% | Recall@20% |
|-------|------------|----------------|------------|
| Logistic Regression | 0.812 | 0.58 | 0.42 |
| Random Forest | 0.841 | 0.63 | 0.48 |
| **Gradient Boosting** | **0.852** | **0.66** | **0.51** |

## Cost-Benefit Analysis

| Scenario | Cost | Benefit | Net |
|----------|------|---------|-----|
| No model (no retention) | $0 | $0 | $0 |
| Random targeting (20%) | $70,430 | $0 | -$70,430 |
| **Model targeting (20%)** | **$70,430** | **$178,500** | **+$108,070** |

Assumptions: CLV = $500, offer cost = $50, retention rate = 10%.

## What Was Learned

- **Tenure is the strongest predictor** — churn drops sharply after 12 months.
- **Month-to-month contracts** churn at 3x the rate of 1-year contracts.
- **Fiber optic** customers churn more (likely due to price sensitivity).
- **Electronic check** payment method correlates with churn (demographic signal).
- Cost-benefit framing makes the model's value tangible to business stakeholders.

## Failure Cases

- Model misses churners who leave due to service quality (not captured in features).
- False positives are mostly new customers (< 3 months) who are still exploring.
- Senior citizens with fiber optic are overpredicted as churn (they actually stay longer).
