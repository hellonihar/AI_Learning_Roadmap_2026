# Manufacturing Fault Detection — Results

## Metrics

| Method | Recall | Precision | F1 | Alarms/Shift |
|--------|--------|-----------|----|-------------|
| Static 3σ thresholds (baseline) | 0.72 | 0.10 | 0.18 | 210 |
| Adaptive EWMA + rolling MAD 3σ | 0.83 | 0.22 | 0.35 | 98 |
| Adaptive + XGBoost (rolling features) | **0.94** | **0.38** | **0.54** | 42 |
| Ensemble (adaptive + XGBoost) | 0.91 | 0.41 | 0.56 | **28** |

## What Was Learned

- Adaptive thresholds (EWMA baseline + MAD-based sigma) cut false alarms by 60% vs. static thresholds
- Sensor drift over time is the #1 cause of false positives — static thresholds become useless after 2 weeks
- Only 38 of 148 sensors carried meaningful fault signal; the rest were noise
- Rolling window features (std over last 10 wafers) were the single strongest predictor of imminent faults

## Failure Cases

- **Gradual degradation** — Slowly drifting sensors (2-3σ over hours) don't trigger adaptive thresholds
- **Batch-start transients** — First 3 wafers after chamber cleaning have different sensor profiles and are always flagged
- **Rare fault types** — Only 2 examples of chemical contamination faults in 3 years; impossible to generalize
