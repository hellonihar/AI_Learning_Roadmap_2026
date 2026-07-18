# Choosing the Right Metric for the Problem

## Why Accuracy Is Often a Bad Metric

Accuracy assumes **equal cost** for all errors. In real problems, different errors have very different costs.

| Problem | Cost of False Positive | Cost of False Negative |
|---|---|---|
| Cancer screening | Mild (extra biopsy) | **Fatal** (missed cancer) |
| Spam filter | **High** (lost important email) | Low (annoyance) |
| Fraud detection | Moderate (customer annoyed) | **High** (money lost) |
| Self-driving | Low (false brake, jarring) | **Catastrophic** (hit pedestrian) |

## Choosing the Right Metric

| Problem | Primary Metric | Why |
|---|---|---|
| **Medical screening** | Recall | Missing a disease is far worse than false alarm |
| **Spam detection** | Precision | Losing important email is worse than seeing spam |
| **Imbalanced dataset** | F1, Precision-Recall AUC | Accuracy is misleading |
| **Search ranking** | Precision@k, NDCG | Users only care about top results |
| **Fraud detection** | Precision@k (top-k alerts) | Investigators have limited capacity |
| **Balanced classes** | Accuracy / AUC-ROC | Simple and meaningful |

## Examples

1. **Class imbalance (fraud detection)**: 99.9% legitimate, 0.1% fraud. Accuracy can be 99.9% by predicting "not fraud" always. Better metric: **Precision-Recall AUC** or **F1-score**. Or business metric: **dollars saved** by catching fraud vs cost of false alarms.
2. **Search ranking**: Showing 10 results for a query. User only looks at top 3-5. Accuracy on all 10 doesn't matter. Metric: **Precision@5** ("of the top 5 results, how many are relevant?"). Or **NDCG** (ranking-aware — placing a relevant result at position 1 is better than position 5).
3. **Customer churn prediction**: Marketing team can reach out to 1000 at-risk customers per month. Model must rank customers by churn probability. Metric: **Precision@1000** ("of the top 1000 flagged, how many actually churned?") — because the business constraint is the outreach capacity.
