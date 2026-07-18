# User Behavior Analytics — Deployment Notes

## Performance
- Sessionization of 2M events: ~45s (pandas groupby/apply)
- Funnel/cohort analysis: ~5-10s
- Dashboards are pre-computed (not live) — refresh daily
- Cost: ~$2/month in BigQuery compute

## Refresh Cadence
- **Daily** — Sessionization + funnel update (last 30 days rolling)
- **Weekly** — Cohort retention tables updated every Monday
- **Monthly** — Engagement score re-computation + segment assignment

## Monitoring
- Track daily: active users, funnel conversion rate, signup-to-activation time
- Alert if signup conversion drops >5% in one day (indicates landing page or auth issue)
- Monitor bot traffic % — rising bot rate inflates engagement metrics
- Dashboard: weekly retention heatmap (cohort week × week number) for product team

## Integration
- Looker dashboard: funnel, cohorts, engagement segments — refreshed daily
- Engagement scores pushed to customer DB for CRM segmentation
- Product team uses drop-off analysis to prioritize onboarding improvements
- Automated alerts via PagerDuty when funnel metrics deviate >2σ
- Amplitude/Mixpanel integration for real-time event analytics (pipeline feeds both)
