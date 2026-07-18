# Customer Churn — Problem Statement

## Business Context
A telecom company loses 15% of customers annually. The cost of acquiring a new customer is 5x the cost of retaining an existing one. The goal is to identify customers likely to churn so retention teams can intervene with targeted offers.

## Problem Type
Binary classification — predict churn (1) or not (0).

## Success Metrics
- **AUC-ROC** ≥ 0.85 — rank-order churn risk.
- **Precision@top20%** ≥ 0.60 — retention team has limited capacity.
- **Cost-benefit**: savings from retained customers must exceed offer cost.

## Constraints
- Must handle imbalanced classes (~73% non-churn, 27% churn).
- Model must be interpretable for retention team to act.
- Inference must be fast (< 50 ms) for real-time scoring during customer calls.
