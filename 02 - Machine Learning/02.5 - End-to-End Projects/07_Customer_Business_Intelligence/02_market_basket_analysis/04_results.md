# Market Basket Analysis — Results

## Metrics

| Metric | Value |
|--------|-------|
| Total frequent itemsets | 1,284 (support ≥ 0.01) |
| Total association rules | 3,472 (lift ≥ 1.2) |
| High-confidence rules (conf ≥ 0.5, lift ≥ 2.0) | 214 |
| Basket coverage of top-10 rules | 34% of multi-item baskets |

## Top Rules (examples)

| Antecedent | Consequent | Support | Confidence | Lift |
|------------|-----------|---------|------------|------|
| Whole milk | Bread | 0.028 | 0.58 | 3.2 |
| Pasta | Tomato sauce | 0.024 | 0.52 | 4.1 |
| Beer | Diapers | 0.019 | 0.45 | 3.8 |
| Yogurt | Berries | 0.022 | 0.49 | 3.5 |
| Coffee | Sugar | 0.031 | 0.42 | 2.8 |

## What Was Learned

- Lift is the critical metric — support drives frequency but lift drives business value
- Many strong rules are obvious (pasta → sauce) but low-lift; the non-obvious ones (beer → diapers) have higher lift
- Filtering to antecedent_len=1 is essential for actionable placement recommendations
- Rules vary significantly by day of week (weekend beer-snack vs. weekday milk-bread)

## Failure Cases

- **Commodity items** — Salt, sugar, flour appear in almost every rule as consequent; low information value
- **Seasonal rules** — BBQ sauce → charcoal has lift 6.2 in July, 1.1 in December; static rules miss this
- **Promotional distortion** — Buy-one-get-free on chips inflates chip-dip lift 2x during promo period only
