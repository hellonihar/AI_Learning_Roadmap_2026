# Dataset — Sales Forecasting

## Source
[M5 Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) (Walmart, Kaggle, 2020). Subset: 3,049 products (foods, household, hobbies) × 10 stores × 1,941 days.

## Size
- **Rows:** ~59 million (full) — but this project downsamples to **store-level aggregates**
- **Time range:** 2011-01-29 to 2016-06-19 (1,941 days, aggregated to ~280 weeks)
- **Features:** 13 raw + engineered calendar features

## Key Columns
| Column | Description |
|--------|-------------|
| `id` | Product + store key |
| `item_id` | Product category (e.g., FOOD_1_001) |
| `store_id` | Store (CA_1, TX_2, ...) |
| `sell_price` | Weekly price (promotions visible here) |
| `d_1 ... d_1941` | Daily sales (wide format originally) |

## Hierarchical Groups
```
Category (3) → Department (7) → Product (3,049)
                    ×
State (3) → Store (10)
```

## Known Challenges
- **Zero-inflated** — many product × week combos have 0 sales (intermittent demand)
- **Promotional distortion** — 30% of weeks have a price discount; demand spikes 2–5×
- **Calendar effects** — Easter, Super Bowl, Thanksgiving shift weekly patterns
- **SNAP benefits** — food sales spike at start of month (not aligned with week)
- **Temporal leakage** — train/test must be time-based, never random shuffle
