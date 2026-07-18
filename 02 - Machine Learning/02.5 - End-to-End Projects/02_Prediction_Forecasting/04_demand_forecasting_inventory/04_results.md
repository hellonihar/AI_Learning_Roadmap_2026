# Results — Demand Forecasting & Inventory

## Benchmark Scores

| Model | MAE (SKU-store-week) | RMSE | Service Level (sim) |
|-------|---------------------|------|---------------------|
| Naïve (last week) | 8.2 | 14.5 | 89% |
| ARIMA (per SKU) | 5.8 | 10.1 | 93% |
| XGBoost (lag features) | 4.1 | 7.6 | 96% |
| XGBoost + ARIMA blend | **3.6** | **6.8** | **98%** |

## Winning Model — XGBoost + ARIMA Blend
- ARIMA captures the slow-moving seasonal baseline
- XGBoost handles promotion spikes and non-linear store effects
- Blend compensates for XGBoost's weakness on smooth periods (overfits to noise) and ARIMA's weakness on sudden lifts

## Business Impact (Simulated)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Inventory Carrying Cost | $4.2M/yr | $3.1M/yr | −26% |
| Stockout Rate | 4.8% | 1.6% | −67% |
| Safety Stock (avg) | 14 units | 9 units | −36% |

## What Was Learned

1. **Bottom-up is better** — reconciliation from SKU-store up to category beats top-down (error reduced 22%)
2. **Demand ≠ Sales** — when out-of-stock, sales undercount true demand; training on sales introduces bias
3. **Promotion pre/post dips** — demand before a promo dips (consumers wait), and demand after dips (stockpiling)
4. **Weekly > daily** — daily demand has too much noise; weekly aggregation stabilises the signal

## Limits
- Perishable goods have unique patterns (markdown before expiry) not well captured
- Cold-start products rely on category-average lags — error 2× higher for first 4 weeks
