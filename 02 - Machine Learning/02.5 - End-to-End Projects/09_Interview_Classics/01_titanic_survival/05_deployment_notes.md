# Deployment Notes — Titanic Survival

## Inference
- Wrap feature engineering + model in a single `Pipeline` with `FunctionTransformer`.
- FastAPI endpoint: `POST /predict` accepts JSON with raw passenger fields.
- Return `{"survival_probability": 0.87, "prediction": 1}`.

## Retraining
- Historical dataset — no retraining needed.
- For production use (e.g., evacuation planning), retrain with updated passenger demographics.

## Monitoring
- Monitor feature distribution drift (Pclass, Sex, Age).
- Log SHAP explanations for audit.

## Interview Talking Points
- "I extracted titles from names because they encode age, gender, and social status in one feature."
- "I imputed Age by Title median rather than global median because Mr vs. Master have very different age distributions."
- "Cabin deck is informative even with 77% missing — the missing indicator itself is predictive (no cabin = lower class)."
- "I used stratified 5-fold CV to account for the 62/38 class split."
