# Insurance Claim Anomaly — Results

## Metrics

| Method | ROC AUC | Precision@Top 5% | Recall@Top 5% |
|--------|---------|-------------------|---------------|
| Modified Z-score | 0.71 | 0.08 | 0.12 |
| Isolation Forest | 0.78 | 0.14 | 0.21 |
| Autoencoder | 0.82 | 0.18 | 0.27 |
| Ensemble (rank-avg) | **0.85** | **0.22** | **0.33** |

## What Was Learned

- Autoencoder outperformed because it learns complex multi-feature interactions (e.g., code+amount+frequency combo fraud)
- Ensemble rank-averaging consistently beat any single method — 3-4% AUC gain
- Modified Z-score alone catches simple amount-based fraud but misses procedural fraud (correct amount, wrong code)
- Reconstruction error per feature shows which aspects of a claim are anomalous (interpretable output for auditors)

## Failure Cases

- **New provider fraud** — Providers with no history have no baseline; their claims get low anomaly scores despite being fraudulent
- **Legitimate expensive procedures** — Organ transplants, cancer treatments have legitimately high costs and get flagged as false positives
- **Billing system errors** — Systematic encoding errors (e.g., wrong default modifier) look like fraud patterns
