# Dataset — Demand Forecasting & Inventory

## Source
Synthetic retail dataset (based on [Favorita](https://www.kaggle.com/c/favorita-grocery-sales-forecasting) Kaggle) + internal POS data structure.

## Size
- **Rows:** ~1.5M (50 stores × 5,000 SKUs × ~6 weeks of daily data, aggregated weekly)
- **Time range:** 104 weeks (2 years)
- **Features:** 18

## Feature Schema
| Group | Features |
|-------|----------|
| Temporal | `week`, `month`, `quarter`, `year`, `is_holiday`, `week_of_month` |
| Product | `category_id`, `perishable`, `avg_price`, `promo_discount` |
| Store | `store_id`, `store_cluster`, `city_tier`, `foot_traffic` |
| Demand | `lag_1`, `lag_2`, `lag_4` (prior weeks demand), `rolling_mean_4` |
| External | `unemployment_rate` (monthly, city-level), `temperature` |

## Target
`units_demanded` — integer count of units ordered/sold that week.

## Known Challenges
- **Intermittent demand** — 40% of SKU-store combos have zero demand in any given week
- **Promotion noise** — 15% price cut can lift demand 3–10×, creating outlier weeks
- **Product lifecycle** — new SKUs (cold start) have 0 history; discontinued SKUs waste space
- **Cross-product cannibalisation** — promo on Brand A steals from Brand B (not captured in single-SKU model)
