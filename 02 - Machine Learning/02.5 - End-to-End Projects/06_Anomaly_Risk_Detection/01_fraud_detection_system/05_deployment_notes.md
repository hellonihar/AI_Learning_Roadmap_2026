# Fraud Detection — Deployment Notes

## Performance
- XGBoost model size: 85 MB (prune to 40 MB by removing trees with gain < 0.001)
- Inference: ~8ms per transaction on CPU (8 cores). Acceptable.
- Batch scoring: 100K transactions in 3.2s

## Retraining Strategy
- **Daily retrain** on last 60 days of labeled data (label delay ~48h after chargeback window closes)
- **Online learning** with XGBoost's `model.boost()` for intra-day updates if fraud spike detected
- Monitor PSI (Population Stability Index) on feature distributions; retrain if PSI > 0.2

## Monitoring
- Track daily: fraud rate, average prediction score, review queue size
- Alert if precision drops below 30% or recall below 50%
- Log all model inputs + predictions for post-hoc analysis
- A/B test new model versions against current champion on 5% traffic for 1 week

## Infrastructure
- REST API endpoint behind load balancer, Redis for feature caching (card aggregates)
- Feature store for card/email aggregates updated every 15 minutes via streaming
