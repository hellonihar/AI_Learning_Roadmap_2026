# Market Basket Analysis — Deployment Notes

## Performance
- Apriori on 20K baskets × 500 items: ~8s (mlxtend implementation)
- Rule generation from itemsets: ~0.5s
- For Instacart-scale (3M baskets): use FP-Growth (mlxtend supports it) for ~5x speedup

## Update Cadence
- **Monthly** — Re-mine rules on rolling 13 months of transaction data
- **Seasonal rules** — Generate separate rule sets for Summer, Holiday, and Baseline seasons
- **Promotion-aware** — Exclude promotional periods from training data, or tag rules with promo flags

## Monitoring
- Track rule stability: what fraction of top-50 rules persist month-over-month
- Monitor average basket size — if it drops, rules may be stale
- A/B test placement changes: compare basket value in test vs. control stores over 4 weeks

## Integration
- Rules stored in PostgreSQL for BI tool access (Looker, Tableau)
- Store layout team receives monthly PDF report: "Top 10 cross-sell placements to adjust"
- Digital: e-commerce site uses rules for "Frequently bought together" widget
- Real-time: checkout screen shows "Customers also bought" at POS
