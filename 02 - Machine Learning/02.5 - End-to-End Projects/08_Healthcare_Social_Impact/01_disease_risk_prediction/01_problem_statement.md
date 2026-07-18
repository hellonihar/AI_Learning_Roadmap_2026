# Disease Risk Prediction — Problem Statement

## Business Context
A hospital wants to identify patients at high risk of heart disease during routine checkups. Early detection enables preventive interventions (lifestyle coaching, medication) and reduces long-term care costs. The model must be interpretable so clinicians trust and act on its predictions.

## Problem Type
Binary classification — predict presence (1) or absence (0) of heart disease.

## Success Metrics
- **Sensitivity (Recall)** ≥ 0.85 — missing a sick patient is costly.
- **Specificity** ≥ 0.80 — avoid overwhelming healthy patients with follow-ups.
- **AUC-ROC** ≥ 0.90 — overall rank-ordering ability.
- **SHAP explanations** required per prediction for clinical audit.

## Constraints
- Model must be explainable (no black-box deep learning).
- Inference < 100 ms per patient.
- Must handle partial missing lab results gracefully.
