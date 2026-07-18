# Customer Segmentation — Deployment Notes

## Performance
- Clustering 50K customers: ~200ms (K-Means with K=5 on RFM data)
- PCA fit + transform: ~80ms
- Full pipeline (RFM calculation + clustering): ~30s (SQL aggregation is the bottleneck)

## Cadence
- **Daily** — Recalculate RFM features from warehouse, re-assign customers to nearest centroid (KMeans.predict)
- **Weekly** — Re-fit K-Means on full dataset (centroids may drift as customer behavior changes)
- **Monthly** — Re-run silhouette analysis to verify K=5 is still optimal

## Monitoring
- Track % of customers per segment over time; sudden shift signals product or market change
- Alert if "Lost" segment grows >5% in one week
- Monitor average Monetary per segment — if Best Customers' spend drops, investigate pricing/competition

## Integration
- Segment labels written to customer database daily
- Marketing automation tools (Mailchimp, HubSpot) pull segment via API for campaign targeting
- A/B test framework: random 10% holdout receives generic campaigns, rest receive segment-targeted campaigns
