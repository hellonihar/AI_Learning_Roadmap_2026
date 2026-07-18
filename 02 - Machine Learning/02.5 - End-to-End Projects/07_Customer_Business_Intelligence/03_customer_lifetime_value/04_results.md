# Customer Lifetime Value — Results

## Metrics

| Model | MAE | MAE (% of avg spend) | Spearman ρ |
|-------|-----|----------------------|------------|
| Mean baseline | $58.40 | 100% | — |
| LinearRegression (RFM) | $42.10 | 72% | 0.52 |
| RandomForest (RFM) | $38.90 | 67% | 0.61 |
| BG/NBD + Gamma-Gamma | **$31.20** | **53%** | **0.73** |

## What Was Learned

- BG/NBD naturally handles zero-repeaters through the dropout probability parameter (p ~ 0.47 means ~47% of customers never repeat)
- Gamma-Gamma requires frequency > 0 — for one-time buyers, use population median monetary value
- Calibration plot: predicted LTV decile vs. actual spend shows near-monotonic relationship (model ranks well)
- penalizer_coef=0.1 improved Gamma-Gamma convergence without degrading fit (some customers have frequency=1 and extreme monetary values)

## Failure Cases

- **High-value one-timers** — Customers who made one very large purchase and never returned; model overestimates future CLV
- **Subscriptions mistaken for repeat** — Monthly recurring charges look like repeat purchases but are automatic; inflates frequency
- **New customers** — Customers with only 2-3 days of history have unreliable frequency → recency estimates; CLV predictions are noisy first 30 days
