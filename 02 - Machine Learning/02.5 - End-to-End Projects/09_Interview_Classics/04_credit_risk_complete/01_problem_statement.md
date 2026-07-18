# Credit Risk — Problem Statement

## Business Context
A bank needs to assess the default risk of loan applicants. A scorecard model assigns a credit score based on applicant characteristics, enabling automated loan approval decisions. The model must be interpretable, auditable, and compliant with banking regulations (Basel, IFRS 9).

## Problem Type
Binary classification — predict default (1) or no default (0).

## Success Metrics
- **AUC-ROC** ≥ 0.80 — rank-order risk.
- **KS statistic** ≥ 0.40 — separation between good and bad.
- Model must produce a **scorecard** (points per feature level) for regulatory review.

## Constraints
- Must use **Weight of Evidence (WoE)** binning for interpretability.
- Must produce a **scorecard** (not just a black-box probability).
- Must include **SHAP explanations** for individual decisions.
- Must document model for regulatory compliance (Basel/IFRS 9).
