# Customer Lifetime Value — Dataset

## Source
[CDNOW dataset](http://www.brucehardie.com/datasets/) (Bruce Hardie) — the standard benchmark for CLV modeling. 2,357 customers over 1997-1998.

## Size & Shape
- **Total**: 69K transactions, 6 columns
- **Columns**: customer_id, date, amount, (plus computed: days_since_first_purchase, cumulative_transactions, cumulative_spend)
- **Time range**: Jan 1997 – Jun 1998 (18 months)
- **Calibration period**: First 12 months (for fitting)
- **Holdout period**: Last 6 months (for evaluation)

## Challenges
- **Zero repeaters** — 42% of customers made only one purchase; BG/NBD must handle this
- **Long tails** — A few customers with 20+ transactions; log-normal distribution for monetary
- **Non-contractual setting** — No explicit churn signal; customers simply stop buying
- **Future spend cap** — Two customers account for 8% of total revenue; model should not extrapolate from outliers
