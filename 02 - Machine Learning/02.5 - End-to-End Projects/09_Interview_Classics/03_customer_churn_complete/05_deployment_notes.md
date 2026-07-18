# Deployment Notes — Customer Churn

## Inference
- Real-time scoring via REST API: `POST /churn-score` → `{churn_probability, risk_tier, suggested_offer}`
- Risk tiers: Low (< 0.3), Medium (0.3–0.6), High (> 0.6)
- Suggested offers by tier:
  - High: retention call + discount
  - Medium: email campaign
  - Low: no action

## Retraining
- Retrain monthly with new churn data.
- Monitor for concept drift: if AUC drops > 0.03, trigger retraining.
- Re-evaluate cost-benefit assumptions quarterly (CLV, offer cost, retention rate).

## Monitoring
- Track AUC-ROC and precision@k weekly.
- Monitor feature distributions (tenure, contract type, monthly charges).
- Log prediction drift (PSI) — if score distribution shifts, retrain.

## A/B Testing Framework
- Randomly assign 10% of customers to "model-guided" retention vs. "business-as-usual."
- Compare churn rate, retention cost, and customer satisfaction.
- Minimum 2-week experiment with 10k customers per arm.
