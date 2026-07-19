# 05.2.4 Monitoring

## Why Monitor ML Systems?

Traditional software monitoring tracks uptime, latency, error rates, and resource utilization. ML monitoring adds a critical dimension: **model behavior**. A model can be up, fast, and returning valid-looking predictions while silently degrading because the data distribution shifted, a feature broke, or the relationship between inputs and targets changed.

Monitoring detects these issues before they cause business impact.

## Types of Drift

### 1. Data Drift (Covariate Shift)

Data drift occurs when the distribution of input features changes between training and production.

**Example**: A fraud detection model trained on 2023 transaction patterns is applied in 2024, where pandemic-era spending patterns have normalized. Feature distributions for `transaction_amount`, `merchant_category`, and `time_since_last_purchase` have all shifted.

**Detection**: Compare production feature distributions against training reference distributions using statistical tests:

| Test                    | Use Case                            |
| ----------------------- | ----------------------------------- |
| Kolmogorov-Smirnov (KS) | Continuous features                 |
| Chi-squared             | Categorical features                |
| Population Stability Index (PSI) | Broad distribution shift   |
| Jensen-Shannon Divergence       | Multi-modal distributions |

```python
from scipy.stats import ks_2samp

stat, p_value = ks_2samp(training_feature, production_feature)
if p_value < 0.05:
    print("Data drift detected!")
```

### 2. Concept Drift (Prior Probability Shift)

Concept drift happens when the relationship $P(y | X)$ changes — the same inputs now map to different outputs.

**Example**: A credit scoring model learns that "has a car loan → low risk" during a stable economy. During a recession, car loan holders default more often. The model's decision boundary is now wrong.

**Subtypes**:
- **Sudden** — regulatory change flips the mapping overnight.
- **Gradual** — consumer behavior slowly shifts over months.
- **Recurring** — seasonal patterns (holiday shopping) that repeat.
- **Incremental** — the boundary drifts step-by-step.

**Detection**: Concept drift requires ground-truth labels, which are often delayed. Techniques include:
- Tracking **model performance** (accuracy, precision) when labels arrive.
- Monitoring **prediction distribution** — if the model predicts class A 60% of the time in training but 40% in production, something changed.
- Using **Drift Detection Methods** (DDM, EDDM, ADWIN) on streaming error rates.

### 3. Prediction Drift

The distribution of model predictions shifts even without ground truth labels. This is a **leading indicator** of potential problems.

```python
# Track the fraction of predictions in each class
expected_distribution = {"approved": 0.7, "denied": 0.3}
current_distribution = {"approved": 0.55, "denied": 0.45}
# PSI > 0.2 indicates significant drift
```

### 4. Model Performance Decay

When ground-truth labels eventually arrive (e.g., loan repayment after 30 days), compute actual metrics:

- Accuracy / F1-score
- Precision / Recall at various thresholds
- RMSE / MAE for regression
- Business metrics (conversion rate, revenue per user)

Set **performance thresholds** that trigger alerts: "If F1 drops below 0.85, page the ML team."

## Monitoring Tools

### Evidently AI

Open-source Python library that computes drift reports and test suites. Integrates with MLflow, Airflow, and monitoring dashboards.

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=training_df, current_data=production_df)
report.save_html("drift_report.html")
```

### WhyLabs / Whylogs

Whylogs is the open-source logging library; WhyLabs is the SaaS platform for monitoring.

```python
import whylogs as why

profile = why.log(production_batch)
profile_view = profile.view()
profile_view.to_pandas()  # summary statistics
```

WhyLabs provides out-of-the-box dashboards for data quality, drift, and performance.

### Arize AI

Full-stack observability platform with tracing (model inputs → predictions → outcomes). Supports computer vision (embedding drift) and LLM monitoring.

### Prometheus + Grafana

For teams that want to build custom monitoring. Export model metrics (prediction count, average confidence, drift scores) as Prometheus metrics, then visualize in Grafana.

```python
from prometheus_client import Histogram, Counter, Gauge, start_http_server

prediction_latency = Histogram("prediction_latency_seconds", "...")
prediction_counter = Counter("predictions_total", "...", ["model_version"])
drift_gauge = Gauge("feature_drift_psi", "...", ["feature_name"])

@prediction_latency.time()
def predict(features):
    prediction_counter.labels(model_version="v1.2").inc()
    ...
```

## Alerts and Dashboards

A good monitoring setup has three tiers:

| Tier    | Audience      | Example Alert                                      |
| ------- | ------------- | -------------------------------------------------- |
| Debug   | ML Engineer   | "Feature `age` PSI = 0.35 > 0.2 threshold"        |
| Warning | ML Team channel | "Data drift detected in 3/10 features"           |
| Critical | On-call      | "F1 score dropped from 0.92 to 0.71"              |

### Setting Thresholds

- **Statistical thresholds**: p-value < 0.05 for KS test, PSI > 0.2.
- **Business thresholds**: Model accuracy falls below 90%, or revenue-per-prediction drops 5%.
- **Volume thresholds**: Predictions drop 50% (feature pipeline might be broken).

Avoid alert fatigue — tier alerts and use **windows** (e.g., "drift detected in 3 of the last 5 batches" rather than on every batch).

## Practical Monitoring Architecture

```
Production Data ─► Feature Store ─► Model ─► Prediction
                                              │
                                              ▼
                                       Log All Features
                                       + Prediction + Metadata
                                              │
                                              ▼
                                       Monitoring Store
                                       (S3 + Parquet)
                                              │
                                              ▼
                         ┌────────────────────┼────────────────────┐
                         ▼                    ▼                    ▼
                   Drift Detection      Performance (when      Data Quality
                   (Evidently)          labels arrive)          (whylogs)
                         │                    │                    │
                         └────────────────────┼────────────────────┘
                                              ▼
                                         Alert Manager
                                              │
                                    ┌─────────┴──────────┐
                                    ▼                    ▼
                              Slack / PagerDuty     Grafana Dashboard
```

## Summary

Monitoring is what separates deployed toys from production systems. Track four forms of degradation — data drift, concept drift, prediction drift, and performance decay — using tools like Evidently, whylogs, and Prometheus. Establish tiered alerts so engineers aren't woken for harmless fluctuations but respond immediately when model quality degrades.
