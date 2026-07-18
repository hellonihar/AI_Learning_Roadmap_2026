# Deployment Notes — Demand Forecasting & Inventory

## Architecture

```
POS Data → Feature Pipeline → Model Inference → Forecast Storage
                                        ↓
                              Replenishment Recommender → ERP
```

## Inference & Latency

| Step | Target | Actual |
|------|--------|--------|
| Feature generation (5K SKUs × 50 stores) | 30 s | 18 s |
| XGBoost predict (all combos) | 10 s | 4 s |
| ARIMA warm-start (9,000 series) | 5 min | 3 min |
| Safety stock calculation | 10 s | 5 s |
| **Total** | < 10 min | < 5 min |

## Retraining Strategy

```python
# Incremental: retrain XGBoost weekly, ARIMA monthly
# XGBoost: append latest week, refit with early stopping
# ARIMA: refit full series on 1st of every month
```

| Model | Frequency | Data Window |
|-------|-----------|-------------|
| XGBoost | Weekly (Sun) | Last 104 weeks |
| ARIMA | Monthly (1st) | Full history |
| Full pipeline | Quarterly | Re-validate feature selection |

## Monitoring

| Metric | Tool | Threshold | Action |
|--------|------|-----------|--------|
| MAE by category | Grafana | > 6 units | Retrain category model |
| Service level (actual) | ERP sync | < 96% | Flag to replenishment team |
| Forecast bias | Weekly report | |bias| > 10% | Re-calibrate safety stock |
| Feature distribution | Evidently | Drift > 0.2 | Investigate promo/price changes |

## Reconciliation
- Bottom-up forecasts reconciled with Top-Down via MinT (Minimum Trace) optimal combination
- Ensures store-level forecasts sum to chain total (needed for central purchasing)

## Key Risks
- **Promotion calendar changes** — new promo types invalidate historical pattern
- **Supply shocks** — supplier disruption means demand goes unmet → training signal corrupted
- **Cold start for new SKUs** — fallback: category-average + store-average blended forecast
