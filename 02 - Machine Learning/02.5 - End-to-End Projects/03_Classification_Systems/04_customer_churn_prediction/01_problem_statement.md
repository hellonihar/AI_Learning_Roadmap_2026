# Problem Statement — Customer Churn Prediction

## Business Context
A subscription SaaS platform loses 5–7% of customers monthly. Reducing churn by 2 percentage points directly increases LTV by 15%. The retention team needs a daily list of high-risk customers for targeted outreach (discounts, check-in calls, feature education).

## Problem Type
Binary classification: active (0) vs. churned (1) at the end of the observation window.

## Success Metrics
- **Recall ≥ 0.85** — catching at-risk customers is more important than precision.
- **F1 score ≥ 0.70** — balance recall with reasonable precision.
- **Lift at top decile ≥ 3.0** — top 10% of predictions should contain 3x the base churn rate.
- Model must output churn probability (not just label) for campaign prioritization.

## Constraints
- Prediction per customer per day; 500k customers → 500k daily inferences.
- Feature lag ≤ 24 hours (must use yesterday's data for today's predictions).
- Interpretability preferred but not mandatory — can use tree ensembles.
