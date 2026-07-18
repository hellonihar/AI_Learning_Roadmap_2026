# Problem Statement — Loan Default Prediction

## Business Context
A peer-to-peer lending platform needs to assess borrower risk at application time. Rejecting good applicants loses origination fees; approving bad ones loses principal. Regulators require explainable decisions — every denial must include the reason.

## Problem Type
Binary classification: fully paid (0) vs. defaulted/charged-off (1).

## Success Metrics
- **AUC-ROC ≥ 0.75** — primary metric for ranking risk.
- **Recall at 10% APR** — expected loss coverage at lending rate.
- **SHAP explainability** required for every prediction; must identify top 3 driving features.

## Constraints
- Inference must complete in < 100 ms per application (real-time decision).
- Model must be interpretable — Logistic Regression or tree-based with SHAP.
- No protected attributes (race, gender, ZIP code proxy) in final model.
- Monthly retraining with new loan performance data.
