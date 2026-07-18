# Deployment Notes — Disease Risk Prediction

## Inference
- Accept JSON with 13 features → return `{"risk_score": 0.87, "prediction": 1, "top_factors": [{"feature": "thal", "shap": 0.42}]}`
- Wrap in FastAPI endpoint `/predict`
- Preprocessing (scaler) must be serialized with `joblib` and loaded at startup

## Retraining
- Retrain quarterly or when new patient data shifts feature distributions (monitor PSI).
- Use the same threshold-tuning step on the validation fold.

## Monitoring
- Track prediction drift (PSI on score distribution).
- Log sensitivity / specificity per month — if sensitivity drops below 0.85, alert.
- Monitor missing-rate per feature — if `ca` missing rate spikes, imputation may bias results.

## Model Card
- Approved for **screening support only** — not a diagnostic tool.
- Requires clinician review before any action.
- Validated on Cleveland population — retrain before deploying to other demographics.
