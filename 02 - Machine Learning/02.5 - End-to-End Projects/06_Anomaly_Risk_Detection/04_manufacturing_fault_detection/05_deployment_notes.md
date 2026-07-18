# Manufacturing Fault Detection — Deployment Notes

## Performance
- Edge inference: ~15ms per wafer on Raspberry Pi-class hardware (rolling features + threshold check)
- Full ensemble (including XGBoost): ~80ms on edge, 5ms on server GPU
- < 500ms total from sensor read to alert — meets real-time requirement

## Retraining Strategy
- **Daily retrain** rolling window baselines (EWMA parameters) — fully automated, no human review
- **Monthly retrain** XGBoost on last 6 months of wafers with confirmed labels
- **Ad-hoc retrain** after chamber maintenance events (new sensor calibration curves)
- Maintain per-chamber models (sensor behavior varies across physical tools)

## Monitoring
- Dashboard: fault rate, top-5 sensors contributing to alarms, detection lead time
- Alert if any sensor shows >30% increase in threshold violations (indicates sensor degradation)
- Track alarm-to-confirmation ratio weekly; if < 2:1, investigate
- Log all sensor readings ± 10 min around each alarm for root cause analysis

## Infrastructure
- Edge deployment on each tool's local controller (docker container)
- Central aggregator for cross-tool pattern detection (rare faults visible only across tools)
- MQTT for real-time sensor ingestion; Kafka for batch processing
- Retraining pipeline triggered by Airflow; model registry in MLflow
