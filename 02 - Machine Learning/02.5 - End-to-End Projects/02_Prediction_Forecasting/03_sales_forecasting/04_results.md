# Results — Sales Forecasting

## Benchmark Scores (Store-level, 2016 test weeks)

| Model | MAE | WMAPE | RMSE |
|-------|-----|-------|------|
| Naïve (same as last week) | 412 | 16.8% | 598 |
| Seasonal Naïve (same week last year) | 385 | 15.2% | 552 |
| ARIMA (1,1,1)×(1,1,1,52) | 340 | 13.1% | 490 |
| XGBoost (lag features + calendar) | **268** | **10.4%** | **410** |
| XGBoost + store dummies | **260** | **9.8%** | **398** |

## Winning Model — XGBoost with Lag Features

- ARIMA captures trend but misses promotional lifts
- XGBoost learns non-linear promotion × holiday interactions
- Lag features at 1, 2, 4, 8 weeks encode the autocorrelation structure

## What Was Learned

1. **Promotional price is the strongest predictor** — weeks with > 20% discount see 2.5× baseline sales
2. **4-week rolling mean stabilises noisy SKUs** — especially for products with intermittent demand
3. **Time-series CV is non-negotiable** — random shuffle overestimates performance by 20–30%
4. **Store-specific models beat one global model** — each store has different customer base and promotion pattern
5. **SNAP benefit timing** adds consistent lift to food categories (first week of month)

## Limits
- 4-week-ahead forecasts degrade significantly (error doubles at week 3+)
- New products with no history — cold start handled via category-average baseline
