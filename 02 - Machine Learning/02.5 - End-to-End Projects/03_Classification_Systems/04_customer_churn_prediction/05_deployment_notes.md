# Deployment Notes — Customer Churn Prediction

## Inference Pipeline

```python
def daily_churn_risk(customers_df: pd.DataFrame) -> pd.DataFrame:
    feats = engineer_features(customers_df)
    feats_encoded = pd.get_dummies(feats, columns=categorical_cols)
    scores = model.predict_proba(feats_encoded)[:, 1]
    return customers_df[["customerID"]].assign(
        churn_prob=scores,
        risk_segment=pd.cut(scores, bins=[0, 0.3, 0.6, 1.0],
                            labels=["low", "medium", "high"])
    )
```

- Latency: ~3 minutes for 500k customers (batch, 4 CPU cores).
- Model size: 25 MB (XGBoost, 200 trees).

## Retraining Strategy
- **Weekly**: retrain on last 12 months of data with latest churn labels.
- **Label definition**: churn = cancellation request + 30 days inactive.
- **Feature freshness**: charge trend, support tickets updated daily via ETL.

## Threshold & Campaign Management
- **High risk** (score ≥ 0.6): auto-trigger retention call or push notification.
- **Medium risk** (0.3–0.6): add to email nurture campaign.
- **Low risk**: no action; monitor weekly trend.
- Thresholds reviewed monthly with retention team based on team capacity.

## Monitoring
- **Model**: AUC-ROC on rolling weekly cohorts, PSI on score distribution.
- **Business**: campaign conversion rate, offer redemption rate, retained customer LTV.
- **Data quality**: alert if any feature has > 5% nulls or distribution shift > 0.2 PSI.

## Infrastructure
- Scheduled Airflow DAG runs at 06:00 UTC daily.
- Scores pushed to CRM via API for campaign execution.
- Shadow scoring on existing retention campaigns for continuous validation.
