# Deployment Notes — Sales Forecasting

## Inference Pipeline

```python
def forecast_next_4_weeks(store_id, model, last_8_weeks):
    future_dates = pd.date_range(start="2026-07-18", periods=4, freq="W")
    preds = []
    history = last_8_weeks.copy()
    for dt in future_dates:
        row = build_features(dt, history)
        pred = model.predict(row.reshape(1, -1))
        preds.append(pred[0])
        history = pd.concat([history, pd.Series(pred[0])])
    return list(zip(future_dates, preds))
```

- **Latency:** ~2 ms per SKU; batch of 3,000 SKUs in 6 seconds
- **Schedule:** Runs every Sunday 02:00 ET before weekly buying cycle

## Retraining Strategy

| Cadence | Trigger | Data Window |
|---------|---------|-------------|
| Weekly (automatic) | Every Sunday | Last 2 years |
| On-demand | WMAPE > 15% for 2 consecutive weeks | Add recent 4 weeks |
| Quarterly | Model version refresh | Full history re-train |

## Monitoring

| Metric | Dashboard | Threshold |
|--------|-----------|-----------|
| WMAPE by store | Grafana | > 12% → alert |
| Forecast bias | Check mean error sign | |bias| > 5% → adjust |
| Promotional lift ratio | Compare discount weeks vs non | Ratio < 1.5 → stale model |
| Data freshness | DAG sensor (Airflow) | Missing yesterday's sales → pause |

## Model Storage
- One `xgboost` model per store (10 models)
- Serialised with `joblib`, versioned in S3/Blob with date tag
- Feature schema locked for backward compatibility on rollback

## Key Risks
- **Supply chain disruption** — model trained on normal patterns; COVID-like shock invalidates all lag features
- **Promotion calendar changes** — new promo types not seen in training → under-predicts lift
