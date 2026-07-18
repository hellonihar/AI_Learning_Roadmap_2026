# Problem Statement — Demand Forecasting & Inventory

## Business Context
A mid-size retail chain (50 stores, 5,000 SKUs) runs on 6 weeks of safety stock. They want to reduce inventory carrying cost by 20% while maintaining 98% service level. The core lever: **better demand forecasts at product × store × week granularity**.

## Problem Type
Hierarchical time-series regression. Forecast demand — not sales — at the most granular level (SKU × store), then reconcile up to category and chain level.

## Success Metrics
| Metric | Target | Notes |
|--------|--------|-------|
| **MAE (SKU-store-week)** | ≤ 5 units | Granular accuracy |
| **Service Level** | ≥ 98% | In-stock rate |
| **Inventory Turns** | ≥ 12× | Current: 8.7× (target: 14×) |
| **Stockout rate** | < 2% | Weeks with zero stock |

## Why Hierarchical?
- Store replenishment is ordered at SKU × store level
- Aggregated forecasts (chain level) hide local stockout patterns
- Bottom-up forecasts summed to top give consistent plans
