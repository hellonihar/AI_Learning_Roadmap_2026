# Problem Statement — Sales Forecasting

## Business Context
A national retailer needs accurate weekly sales forecasts for 3,049 products across 10 stores. Forecasts drive inventory procurement, shelf-space planning, and promotional calendar decisions. Even a 1% improvement in forecast accuracy saves $1M+ in waste and stockouts.

## Problem Type
Time-series regression with seasonality. Hierarchical (product × store × week), but this project focuses on **aggregate store-level forecasting** as a starting point.

## Success Metrics
| Metric | Target | Notes |
|--------|--------|-------|
| **MAE** | ≤ 300 units/store/week | Interpretable avg error |
| **WMAPE** | ≤ 12% | Weighted (revenue-scaled) mean abs percentage error |
| **RMSE** | ≤ 500 units | Penalises large weekly misses |

## Data Frequency
- **Weekly** aggregation (sales data, not daily)
- **Forecast horizon:** next 4 weeks (rolling, re-estimated each week)

## Constraints
- Must handle seasonality (holiday spikes, back-to-school)
- Promotional lift must be separable from baseline demand
- Inference on 3,000+ SKUs in under 30 seconds
