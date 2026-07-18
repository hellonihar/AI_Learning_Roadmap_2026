# Deployment Notes — Loan Default Prediction

## Inference Pipeline

```python
def assess_application(app: dict) -> dict:
    df = pd.DataFrame([app])
    proba = model.predict_proba(df)[0, 1]
    decision = "approve" if proba < RISK_THRESHOLD else "review"
    shap_exp = explainer.shap_values(model.named_steps["prep"].transform(df))
    return {"default_prob": proba, "decision": decision, "shap": shap_exp.tolist()}
```

- Latency: ~45 ms per application (preprocessing + LR inference + SHAP).
- Model size: < 1 MB (sparse coefficients).

## Retraining Strategy
- **Monthly cadence**: train on last 24 months of completed loans.
- **Label maturity**: default label is reliable only after 12 months of payment history.
- **Monitoring**: AUC-ROC on rolling monthly cohorts; alert if dropped below 0.70.

## Threshold Tuning
- Risk threshold set by credit committee, not data science.
- Common approach: approve if default prob < 0.15, review if 0.15–0.35, deny if > 0.35.
- Each threshold change requires compliance sign-off.

## Fairness Monitoring
- Track AUC and rejection rate across protected groups (via proxy: ZIP + income bracket).
- If adverse impact ratio exceeds 4/5ths rule, initiate model review.

## Infrastructure
- Deployed as a batch-scoring job (hourly) and real-time API (for instant decisions).
- SHAP explanations stored in audit DB alongside decision.
- Model cards generated on each version deploy — includes training window, AUC, feature list.
