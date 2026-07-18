# Network Intrusion Detection — Deployment Notes

## Performance
- Ensemble model: ~2ms per connection record on CPU
- PCA preprocessing + inference = ~0.5ms for the full pipeline
- Peak throughput: 50K connections/second per node (well above typical enterprise needs)

## Retraining Strategy
- **Weekly retrain** on rolling 7-day window of network traffic
- **Ad-hoc retrain** when new CVE announcements match infrastructure stack
- Keep a held-out set of confirmed attacks for threshold calibration

## Monitoring
- Track flag rate (% flagged as anomalous) per hour per subnet
- Alert if flag rate deviates >2σ from rolling 7-day average
- Dashboards: ROC curves per attack category (updated weekly), FPR vs. time
- Log all flagged connections (src_ip, dst_ip, port, flags, score) for SIEM ingestion

## Integration
- Outputs to Kafka topic consumed by SOAR (Security Orchestration) for automated blocking
- Threshold configurable per subnet (DMZ can tolerate lower threshold than internal LAN)
