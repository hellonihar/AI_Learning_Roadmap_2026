# Deployment Notes — House Price Prediction

## Inference

- **Latency target:** < 100 ms per prediction
- **Format:** Serialise the full sklearn Pipeline with `joblib.dump` — this bundles preprocessing + model into one object
- **Serving:** FastAPI endpoint accepting JSON with raw feature values; returns `{"predicted_price": 285000}`

```python
import joblib
pipe = joblib.load("model/house_price_pipeline.joblib")
pred = pipe.transform(single_row)    # returns log-price
dollars = float(np.expm1(pred))
```

## Retraining Frequency

- **Quarterly** — housing markets shift with interest rates and inventory
- **Trigger:** if RMSE drifts > 15% above baseline in production monitoring
- **Data freshness:** always include last 3 months of closed sales

## Monitoring

| Signal | Tool | Threshold | Action |
|--------|------|-----------|--------|
| Prediction drift | Evidently AI | PSI > 0.15 | Retrain |
| Feature drift (Neighborhood %) | Custom check | > 10% shift | Investigate data |
| Error spikes | Prometheus | RMSE > $35k | Alert + fallback to Ridge model |
| Latency p99 | Grafana | > 500 ms | Scale pods |

## Key Risks
- **Regulatory** — Must not proxy protected attributes. Check feature importance for zip-code proxies
- **Fair lending** — Monitor prediction residuals across demographics; parity test quarterly
