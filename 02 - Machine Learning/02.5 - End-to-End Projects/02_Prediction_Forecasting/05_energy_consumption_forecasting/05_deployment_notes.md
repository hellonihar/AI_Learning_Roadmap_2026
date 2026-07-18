# Deployment Notes — Energy Consumption Forecasting

## Inference Architecture

```
Smart Meter → MQTT → Stream Processor → Feature Store → Model API → Grid Scheduler
                                                              ↓
                                                    Forecast Cache (Redis)
```

## Latency Requirements

| Step | Budget | Actual |
|------|--------|--------|
| Meter read ingestion | 1 min | 10 s |
| Feature computation | 10 s | 3 s |
| Model inference (single household) | 100 ms | 15 ms |
| Batch (10,000 households) | 30 s | 12 s |
| **End-to-end hourly cycle** | < 5 min | < 2 min |

## Retraining Strategy

| Trigger | Action | Cadence |
|---------|--------|---------|
| Seasonal shift | Full retrain | Monthly |
| RMSE drift > 15% | Immediate retrain | On alert |
| New household onboarding | Cold-start → ramp | 2 weeks of data → active model |
| Weather API change | Re-validate feature pipeline | On change |

## Monitoring

```python
# Production drift check
def check_drift(current_preds, actuals, threshold=0.35):
    rmse = np.sqrt(np.mean((actuals - current_preds) ** 2))
    if rmse > threshold:
        alert("Energy forecast RMSE above threshold")
    return rmse
```

| Metric | Tool | Alert |
|--------|------|-------|
| RMSE (hourly) | Grafana | > 0.35 kWh |
| Bias (24h cumulative) | Custom | |bias| > 2 kWh |
| Data freshness latency | Prometheus | Last reading > 2 hours → P1 |
| Model inference latency | Prometheus | p99 > 500 ms |

## Model Versioning
- MLflow registry; each monthly retrain validated against last 4 weeks
- Rollback: keep last 3 production models; switch on drift alert

## Key Risks
- **Smart meter outage** — if 20%+ meters offline, grid load estimate degrades
- **Weather forecast error** — temperature off by 5°C → consumption error doubles
- **Daylight Saving Time** — duplicate/missing hour breaks 24-lag features (handle with explicit UTC)
