# Problem Statement — Credit Card Fraud Detection

## Business Context
A payment processor handles millions of daily transactions. Fraud costs the business chargeback fees, lost merchandise, and reputational damage. However, blocking legitimate transactions (false positives) loses revenue and frustrates customers.

## Problem Type
Binary classification — legitimate (0) vs. fraudulent (1). Extremely imbalanced.

## Success Metrics
- **Precision-Recall AUC (PR-AUC)** — primary metric. Accuracy is useless at 0.1% prevalence.
- **Recall ≥ 0.80** at a precision floor of 0.50 — catch most fraud without overwhelming the review team.
- **F2-score** (recall-weighted) — because missed fraud is costlier than false alarms.

## Constraints
- Inference must complete in < 20 ms per transaction (real-time authorization).
- Model must be retrained daily on rolling 90-day window (fraud patterns shift fast).
- Interpretability is not required (automated block/flag only).
