# Customer Segmentation — Results

## Optimal K: 5 (Silhouette Score: 0.42)

| Segment | Label | % Customers | Recency (days) | Frequency | Monetary (£) |
|---------|-------|-------------|----------------|-----------|-------------|
| 0 | Best Customers | 12% | 8 | 42 | £2,850 |
| 1 | Loyal Regulars | 28% | 18 | 18 | £820 |
| 2 | At Risk | 22% | 90 | 8 | £340 |
| 3 | New/Warm Leads | 25% | 25 | 3 | £95 |
| 4 | Lost/Churned | 13% | 280 | 2 | £60 |

## What Was Learned

- Silhouette score of 0.42 indicates moderate cluster separation — acceptable for a 3D RFM space
- Log-transform was critical: raw Monetary is Pareto-distributed (80/20 rule) and masks all structure
- Segment 2 (At Risk) is the most actionable — high past value but >90 days dormant; re-engagement campaigns should target these
- PCA explained 88% variance with 2 components (Recency dominated PC1, Frequency+Monetary dominated PC2)

## Failure Cases

- **One-time big spenders** — Customers with one very large purchase cluster with "Best Customers" incorrectly
- **B2B customers** — Corporate accounts with bulk orders have extreme frequency values that form a tiny 6th cluster
- **Seasonal buyers** — Holiday-specific shoppers (Christmas decor) look "At Risk" 10 months of the year
