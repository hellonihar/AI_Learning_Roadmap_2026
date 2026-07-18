# Market Basket Analysis — Dataset

## Source
[UCI Online Retail](https://archive.ics.uci.edu/ml/datasets/Online+Retail) (same as segmentation project) — but also cross-reference with [Instacart Market Basket](https://www.kaggle.com/c/instacart-market-basket-analysis) for grocery-specific patterns.

## Size & Shape
- **Online Retail**: 541K invoice lines, ~20K transactions. Per-invoice product list.
- **Instacart**: 3.2M orders, ~50K products, 206K users. Order-level product sets.
- **Transaction format**: One row per product per order (basket view after pivot).

## Challenges
- **Basket size variation** — Single-item purchases (~30%) add noise; filter to basket size >= 2
- **Item cardinality** — 50K+ unique products; many appear in <5 baskets
- **Temporal seasonality** — BBQ items co-occur with beer in summer but not winter
- **Multiple units** — "2x milk" and "1x milk" are the same item; aggregate quantities
