# Customer Lifetime Value — Deployment Notes

## Performance
- BG/NBD fit on 2,357 customers: ~0.5s
- Gamma-Gamma fit: ~0.2s
- Prediction for 100K customers: ~0.3s
- Full pipeline (RFM calc → model fit → predict): ~5s for 100K customers

## Retraining Strategy
- **Monthly** — Re-fit BG/NBD + Gamma-Gamma on expanding window (all historical data)
- **Quarterly** — Validate calibration (lift chart decile analysis) and adjust if degradation >10%
- Stable model parameters over 6+ months indicate mature customer base

## Monitoring
- Track MAPE per decile — degradation in top decile is most costly
- Monitor p (dropout probability) and λ (transaction rate) over time — a rising p means customers are churning faster
- Alert if Spearman ρ drops below 0.6 or MAE exceeds 60%
- A/B test marketing spend allocation: model-based targeting vs. existing rule-based targeting

## Integration
- CLV scores written to customer data warehouse (Snowflake) daily
- Segmentation rules: Top 20% CLV → VIP treatment, Bottom 20% → win-back campaigns
- Ad platforms (Google Ads, Meta) receive CLV segments via CRM sync for bid optimization
- Notebook scheduled via Airflow, outputs to Looker dashboard for marketing team
