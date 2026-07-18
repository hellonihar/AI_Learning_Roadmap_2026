# Results — Energy Consumption Forecasting

## Benchmark Scores

| Model | RMSE (kWh) | MAE (kWh) | R² |
|-------|-----------|-----------|-----|
| Persistence (last hour) | 0.52 | 0.38 | 0.65 |
| Linear Regression | 0.42 | 0.30 | 0.78 |
| Random Forest | 0.33 | 0.24 | 0.86 |
| XGBoost (tuned) | **0.29** | **0.21** | **0.89** |
| LSTM (24h lookback) | 0.31 | 0.22 | 0.88 |

## Winning Model — XGBoost

- Outperformed LSTM on this dataset size (~1.5M rows after cleaning) — tree-based models handle the tabular + time features well
- Training time: 45 s (XGBoost) vs 25 min (LSTM, GPU)
- Feature importance interpretable out of the box

## Error by Hour (RMSE)

| Hour | RMSE | Note |
|------|------|------|
| 0–5 AM | 0.18 | Low base load, predictable |
| 6–9 AM | 0.34 | Morning ramp, high variability |
| 10 AM–4 PM | 0.28 | Moderate |
| 5–9 PM | 0.38 | Evening peak, cooking+heating |
| 10–11 PM | 0.31 | Decreasing |

## What Was Learned

1. **24-hour lag is critical** — correlation of load[t] with load[t−24h] is r = 0.85; without it, RMSE jumps to 0.45
2. **Temperature > hour** — temperature of the current hour is the single strongest feature (r = 0.62 with consumption in winter)
3. **Weekend profiles differ** — morning peak shifts 2+ hours later; separate weekend model reduced RMSE 7%
4. **Sub-metering helps** — the three sub-circuits explain specific appliances; adding them dropped RMSE from 0.33 → 0.29

## Limits
- Single household — does not generalise to industrial or apartment buildings
- Weather data granularity (daily avg) loses intra-day cloud cover effects
