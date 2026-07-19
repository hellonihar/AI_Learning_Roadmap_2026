# 05.2.8 Retraining

## When to Retrain

A model's predictive performance degrades over time as the world changes. Deciding when to retrain is a trade-off between cost (compute time, engineering effort, validation) and benefit (maintaining accuracy).

### 1. Time-Based Retraining

The simplest strategy: retrain on a fixed schedule.

- **Weekly**: For fast-moving domains (fraud detection, recommendation).
- **Monthly**: For moderately stable domains (customer churn, demand forecasting).
- **Quarterly**: For slow-changing domains (credit risk, medical diagnosis).

**Pros**: Predictable resource usage, easy to automate.
**Cons**: Wastes compute if the model is still performing well; misses between-schedule degradation.

```python
# Airflow DAG triggered weekly
with DAG("retrain", schedule_interval="0 6 * * 1"):  # every Monday 6 AM
    ...
```

### 2. Performance-Based Retraining

Trigger retraining when monitored metrics fall below a threshold.

- Online metrics (when ground truth is available): accuracy drops below 0.85.
- Business metrics: conversion rate drops 5% week-over-week.
- Proxy metrics: prediction distribution shifts significantly.

**Pros**: Cost-efficient — retrain only when needed.
**Cons**: Requires ground-truth labels (often delayed), complex alert threshold tuning.

```python
if current_accuracy < 0.85:
    trigger_retraining_pipeline()
```

### 3. Drift-Triggered Retraining

Use drift detection (see 05.2.4 Monitoring) as the trigger.

- **Data drift detected**: Feature distributions have shifted. Retrain on recent data.
- **Concept drift detected**: P(y|x) has changed. May require feature re-engineering or model architecture change.
- **Prediction drift**: Output distribution has shifted. Investigate before retraining — may indicate a data pipeline bug rather than genuine drift.

```python
from evidently import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(ref_data=train_df, curr_data=latest_df)
drift_score = report.as_dict()["metrics"][0]["result"]["drift_score"]

if drift_score > 0.3:
    trigger_retraining()
```

## Retraining Strategies

### Full Retraining

Train a completely new model from scratch on all available data (old + new).

| Pros                                    | Cons                                      |
| --------------------------------------- | ----------------------------------------- |
| Simple to implement                     | Computationally expensive                 |
| No risk of catastrophic forgetting      | Takes longer                              |
| Can incorporate all historical patterns | May overfit if data distribution shifted  |

**Best for**: Small datasets, infrequent retraining, models where all historical patterns are relevant.

### Incremental / Online Learning

Update the existing model with new data without retraining from scratch.

```python
# sklearn: partial_fit for online learning
model.partial_fit(X_new, y_new, classes=[0, 1, 2])

# PyTorch: finetune a few epochs on new data
for batch in new_data_loader:
    loss = criterion(model(batch), batch_labels)
    loss.backward()
    optimizer.step()
```

| Pros                                    | Cons                                      |
| --------------------------------------- | ----------------------------------------- |
| Very fast — seconds instead of hours    | Risk of catastrophic forgetting           |
| Low compute cost                        | Requires model support for partial updates |
| Natural for streaming data              | Harder to validate (no train/test split)   |

**Best for**: High-frequency data streams (ad CTR, real-time fraud), large models where full retraining is prohibitive.

### Windowed Retraining

Train only on the most recent N days/months of data.

- **Sliding window**: Keep a fixed-size window of recent data (e.g., last 90 days).
- **Tumbling window**: Divide time into non-overlapping chunks, train on each chunk.

```python
window_data = df[df.timestamp > (now - timedelta(days=90))]
model = train(window_data)
```

**Best for**: Domains where old patterns actively hurt performance (e.g., post-COVID retail, fashion trends).

## Handling Stale Models

When a model is in production but not recently retrained:

- **Shadow the stale model** — run the new model in shadow mode (see below) to compare.
- **Revert to a simpler baseline** — a linear model trained on fresh data may outperform a stale gradient-boosted model.
- **Detect and alert** — if the stale model's predictions have drifted significantly (no ground truth needed), flag for manual inspection.

## Shadow Deployment

Run the new model in parallel with the current production model without serving its predictions to users.

```
User Request ──► Production Model ──► User-facing Response
                     │
                     ▼
               Shadow Model ──► Log predictions and metrics (no external effect)
```

After N days of shadow data:
1. Compare shadow model's predictions against production.
2. If shadow model is better (more accurate on delayed ground truth, or more confident), promote it.
3. If worse, discard — no user impact.

Shadow deployment is **risk-free testing** for model changes.

## A/B Testing for Models

A/B testing serves different models to different user segments and compares business metrics.

| Variant | Model      | % Traffic | Metric                     |
| ------- | ---------- | --------- | -------------------------- |
| A       | Production | 90%       | Conversion rate (baseline) |
| B       | Candidate  | 10%       | Conversion rate            |

Requires:
- **Splitting**: Consistent assignment (user ID hash → variant).
- **Instrumentation**: Log which variant served each request.
- **Statistical test**: Is the difference significant (p < 0.05)?

```python
# A/B split
if hash(user_id) % 100 < 10:
    prediction = candidate_model.predict(features)
else:
    prediction = production_model.predict(features)
```

A/B testing is the **gold standard** for model validation but requires significant infrastructure and traffic volume.

## Retraining Pipeline Pattern

```
Trigger ──► Data Extraction ──► Validation ──► Preprocessing ──► Training ──► Evaluation ──► Compare
  │                                                                                          │
  │                                                                                          ▼
  └──── (time/drift/performance) ──────────────────────────────────────────────── (rollback or promote)
```

## Summary

Choose a retraining strategy that matches your data's rate of change:

| Strategy       | Speed     | Cost   | Risk           | Use Case                     |
| -------------- | --------- | ------ | -------------- | ---------------------------- |
| Time-based     | Predictable | Medium | May waste compute | Stable domains               |
| Performance    | Reactive  | Low    | Delayed detection | When labels are available    |
| Drift-triggered | Proactive | Low  | Threshold tuning | Fast-moving domains         |
| Full retrain   | N/A      | High   | Low            | Small data, infrequent       |
| Incremental    | Instant   | Very low | Catastrophic forgetting | Streaming data         |
| Windowed       | Medium   | Medium  | Loses old patterns | Time-sensitive domains   |

Validate every new model through shadow deployment or A/B testing before full rollout. Never swap models blindly — measure, compare, and then promote.
