# Feedback and Monitoring

Monitoring is how you know your model is still working after deployment. A model that achieved 0.95 AUC at launch can degrade to random guessing within weeks due to data drift, concept drift, or pipeline failures — and without monitoring, you won't notice until users complain or business metrics collapse. Feedback is how the system improves over time — collecting labels, outcomes, and user signals to retrain and refine the model.

Monitoring and feedback form the closed loop that turns a static model into a learning system. Together they answer: Is the model still accurate? Is production data still compatible? Are we collecting the signals needed to improve? Investing in monitoring and feedback infrastructure upfront pays for itself the first time it catches a silent regression before it affects the business.

## 7.1 Monitoring What Matters

### Data Distribution Monitoring

Monitor the distributions of input features to detect **covariate drift** before it degrades predictions.

| Signal | What It Detects | Tool / Method |
|---|---|---|
| **Feature mean / std / quantiles** | Overall distribution shift | Statistical profiling (pandas-profiling, whylogs, Evidently) |
| **PSI (Population Stability Index)** | Magnitude of shift per feature | PSI > 0.1 = warning, > 0.2 = critical |
| **KS test / JS divergence** | Statistical significance of change | Alerts when p < 0.01 |
| **Null rate per feature** | Data source degradation | Baseline ± 2σ threshold |
| **Unique values / cardinality** | Schema drift or data corruption | Compare to expected range |
| **Min / max range** | Out-of-range values | Validate against expected bounds |

**Common stack:**
- **Python:** Great Expectations, Pandera, Evidently, whylogs
- **Managed:** AWS SageMaker Model Monitor, GCP Vertex AI Model Monitoring, Azure ML Data Drift
- **Streaming:** Deequ (on Spark), Monte Carlo, Soda

**Best practice:** Set up per-feature alerts before deploying the model. When drift fires, don't panic — most drift is benign. Investigate in priority order: (1) concept drift (most dangerous), (2) pipeline changes, (3) benign seasonal shifts.

**Real example:** A recommendation system's `click_7d` feature suddenly dropped 40%. Investigation revealed an upstream logging change that stopped recording clicks for logged-out users. The model's CTR dropped 15% because all logged-out users received default predictions. Monitoring caught it within 2 hours.

### Prediction Distribution Monitoring

Monitor what the model is predicting. Shifts in prediction distribution often signal concept drift or data pipeline issues.

| Signal | What It Detects |
|---|---|
| **Mean prediction score** | Model is predicting higher/lower on average |
| **Prediction std / variance** | Model is more/less confident |
| **Score distribution (histogram)** | Shape change — bimodal → flat, concentrated → spread |
| **Fraction of predictions above threshold** | How many decisions would change |
| **Prediction by segment** | Drift in specific user groups masked by aggregate |

**Example thresholds:**
| Signal | Alert Condition |
|---|---|
| Mean prediction score | > ±5% from baseline in 24h rolling window |
| % predictions > 0.9 | > ±20% relative change |
| % predictions in top decile | > ±10% relative change |
| Null / NaN predictions | Any increase from 0% |

**Common stack:**
- **Dashboard:** Grafana + Prometheus, Datadog, New Relic
- **ML-specific:** WhyLabs, Arize AI, Evidently, NannyML

**Best practice:** Log every prediction (model version, timestamp, score, feature hash). This enables retrospective analysis after any incident: "Did predictions shift before the business metric dropped?"

### Label Quality Monitoring

Labels are ground truth — garbage labels produce garbage models.

| Issue | Detection | Mitigation |
|---|---|---|
| **Label noise** (wrong labels) | Inter-annotator agreement, label consistency checks | Use majority vote, expert review for disagreements |
| **Label drift** (label distribution shifts) | Compare label distribution month-over-month | Retrain if P(Y) shifts significantly |
| **Label delay** (labels arrive weeks late) | Track label arrival time vs prediction time | Use time-windowed training (only labels that arrived) |
| **Missing labels** (no outcome recorded) | Monitor label rate (% of predictions with labels) | Investigate if label rate drops below expected |

**Common stack:**
- **Labeling platforms:** Labelbox, Scale AI, Snorker AI, Prodigy
- **Quality checks:** Agreement metrics (Krippendorff's alpha, Cohen's kappa), gold-standard questions

**Best practice:** Reserve 5–10% of labeled data for gold-standard questions — known-answer questions hidden in labeling tasks. If annotators score < 90% on gold questions, retrain or replace them.

---

## 7.2 Performance Monitoring

### Online Metrics vs Offline Metrics

| | Offline Metrics | Online Metrics |
|---|---|---|
| **When** | Before deployment, on test set | In production, on live traffic |
| **Data** | Fixed, curated test set | Live data with noise, missing values, delays |
| **Labels** | Available immediately | Delayed (minutes to months) |
| **Metric** | AUC, F1, RMSE | Business metrics (CTR, revenue, latency), proxy metrics |
| **Bias** | Low (controlled evaluation) | High (selection bias, confounding) |
| **Purpose** | Does the model work? | Is the model working today? |

**Gap example:**
- Offline F1: 0.92 (on clean, balanced test set)
- Online F1: 0.78 (on live traffic with 30% missing features, skewed distribution)

**The gap between offline and online metrics is the most important number to track.** If it grows, your training distribution is diverging from production.

**Common stack:**
- **Offline:** MLflow, Weights & Biases, Neptune, SageMaker Experiments
- **Online:** Grafana, Datadog, Prometheus, custom dashboards

**Best practice:** Report offline and online metrics side-by-side on the same dashboard. A growing gap triggers investigation before the online metric crosses the failure threshold.

### Alert Fatigue

Too many alerts = no alerts respected. Design your alerting to minimize noise.

| Alert Level | Condition | Response |
|---|---|---|
| **Info** | PSI 0.05–0.1, small metric dip | Log, no notification |
| **Warning** | PSI 0.1–0.2, metric drop > 2% | Page ML team during business hours, create ticket |
| **Critical** | PSI > 0.2, metric drop > 5%, error rate spike | Page on-call immediately, consider rollback |
| **Silent** | Expected seasonal drift (holiday, promotion) | Suppress alert, note in monitoring dashboard |

**Anti-patterns that cause alert fatigue:**

| Anti-pattern | Fix |
|---|---|
| Alert on every PSI > 0.1 | Use tiered alerts; most drift is benign |
| Static thresholds for all features | Per-feature thresholds (some features are naturally volatile) |
| No time-of-day / day-of-week awareness | Include expected patterns (e.g., lower conversion on Sundays) |
| Same alert for model team and business team | Separate operational alerts from business metric reports |

**Real example:** A team set up PSI alerts on all 200 features with a uniform threshold. Day 1: 47 alerts. Day 5: 0 alerts (nobody reads them anymore). They switched to per-feature tiers with seasonality awareness → 2–3 meaningful alerts per week.

### Defining Meaningful Thresholds

A metric moving 0.1% is not actionable. Define thresholds based on **business impact**, not statistical significance.

| Approach | How | Example |
|---|---|---|
| **Business-driven** | How much metric change justifies human investigation? | "Revenue drop > 1% = investigate" |
| **Baseline + σ** | Use historical mean ± 2σ or 3σ | "Mean prediction score > 0.6 (was 0.55 ± 0.02)" |
| **Time-based** | Compare same day last week, last month | "CTR on Sunday is 2.1% — last 4 Sundays averaged 2.3%" |
| **Segment-first** | Overall metric may be stable while a segment tanks | "Mobile users' precision dropped 15% (desktop is flat)" |

**Best practice:** Set temporary thresholds on day 1, review after 2 weeks, adjust. Thresholds that never fire are too loose. Thresholds that fire daily are too tight. Aim for 1–3 actionable alerts per week per model.

**Common stack for observability:**
- **Monitoring + dashboards:** Grafana + Prometheus, Datadog, New Relic, Honeycomb
- **ML-specific:** Arize AI, WhyLabs, Evidently, NannyML
- **Log aggregation:** ELK Stack (Elasticsearch, Logstash, Kibana), Loki, Splunk

---

## 7.3 Feedback Loops

### Explicit vs Implicit Feedback

| | Explicit Feedback | Implicit Feedback |
|---|---|---|
| **What** | User directly rates/says preference | Observed user behavior |
| **Examples** | Thumbs up/down, star rating, survey | Click, dwell time, purchase, scroll depth |
| **Volume** | Low (most users never give explicit feedback) | High (every interaction generates data) |
| **Bias** | Self-selection bias (only engaged users rate) | Behavioral bias (click ≠ like, could be clickbait) |
| **Signal strength** | Strong (user stated intent) | Weak (inferred intent, needs calibration) |
| **Cost** | High (prompting users, friction) | Low (logged automatically) |

**Best practice:** Use implicit feedback as the primary training signal (you have lots of it). Use explicit feedback for calibration and evaluation (it's cleaner but sparse). Blend both with different weights.

**Real example:** A news recommendation system trained on clicks (implicit) optimized for clickbait. Adding "time spent on article" as an implicit signal (implicit, but proxies engagement better than clicks alone) and thumbs-up ratings (explicit, but 0.1% of users provide them) improved content quality scores by 30%.

### Feedback Delay

Feedback (labels, outcomes) often arrives long after the prediction was made.

| Delay | Domain | Challenge |
|---|---|---|
| **Seconds–minutes** | Click prediction, ad CTR | Near-instant feedback — can update quickly |
| **Hours–days** | Content moderation, support ticket routing | Fast enough for daily retraining |
| **Weeks–months** | Churn prediction, loan default | Must wait for label; stale training data |
| **Years** | Lifetime value, drug efficacy | May never see full label; use proxy |

**Strategies:**

| Strategy | How It Works | Trade-off |
|---|---|---|
| **Surrogate labels** | Use early signals as proxy (7-day activity for churn) | Noisy but fast |
| **Time-windowed training** | Train only on samples with observed labels within window | Biased toward shorter-horizon outcomes |
| **Survival analysis** | Model censored data — "still alive" vs "churned" | Complex, needs specialized methods |
| **Delayed feedback modeling** | Weight samples by label arrival probability (often used in ads) | Additional modeling complexity |

**Common stack:**
- **Streaming:** Kafka, Kinesis for collecting delayed feedback
- **Storage:** Feature store / data warehouse for joining predictions with delayed labels
- **Orchestration:** Airflow to schedule delayed label joins

### Feedback Bias

Feedback is not random — it's systematically biased. Using feedback naively perpetuates bias.

| Bias Type | Description | Example | Mitigation |
|---|---|---|---|
| **Position bias** | Items shown higher get more clicks regardless of relevance | Top search result has 30% CTR, bottom has 1% | Use position-aware models, counterfactual evaluation (IPW) |
| **Selection bias** | Only labeled cases have feedback (e.g., only reviewed fraud cases get labels) | Model is evaluated only on flagged cases | Importance weighting, propensity scoring |
| **Exposure bias** | Model only gets feedback on items it showed | Never discovers that user likes a niche genre | Exploration (ε-greedy, Thompson sampling) |
| **Survivorship bias** | Only active users generate feedback | Churned users have no data for retraining | Train on historical data before churn; treat missing = negative |
| **Confirmation bias** | Model predicts X → user sees X → user clicks X → confirms model's belief | Recommender echo chamber | Explicit diversity, counterfactual logging |

**Best practice:** Log the **entire candidate set** that was scored, not just the prediction that was shown. This enables offline counterfactual evaluation (what would have happened if we showed item B instead of A). Tools like Criteo's LV-IPS or Google's TensorFlow Ranking support these methods.

**Common stack for feedback systems:**
- **Experimentation:** A/B testing frameworks (Google Optimize, LaunchDarkly, Eppo)
- **Counterfactual evaluation:** Replay logs with different policies, off-policy evaluation libraries
- **Bandit / exploration:** Vowpal Wabbit, Azure Personalizer, Thompson sampling libraries

---

## 7.4 Model Decay & Retraining Triggers

### Scheduled Retraining

Retrain on a fixed calendar schedule — the simplest and most common approach.

| Frequency | Best For | Cost |
|---|---|---|
| **Daily** | Fast-changing patterns (fraud, news, ads) | High — full train pipeline every day |
| **Weekly** | Most production models (recommender, churn, ranking) | Moderate |
| **Monthly** | Stable patterns (credit scoring, LTV) | Low |
| **Quarterly** | Regulatory models, long-term trends | Very low |

**Best practice:** Even with scheduled retraining, monitor for unexpected decay. If performance drops between scheduled retrains, trigger an unscheduled retrain.

### Event-Based Retraining

Trigger retraining when something changes in the environment.

| Trigger Event | Example | Action |
|---|---|---|
| **Data drift detected** | PSI > 0.2 on critical features | Retrain on recent data |
| **Performance drop** | AUC drops > 0.02 from last eval | Retrain, investigate root cause |
| **Business metric change** | Revenue drop correlated with model rollout | Rollback or retrain |
| **New data available** | Fresh labels arrived for last 30 days | Incorporate into training |
| **Feature change** | New feature added, old feature deprecated | Retrain with updated feature set |
| **Season boundary** | Holiday season ends → back to normal behavior | Retrain on post-season data |

**Common stack:**
- **Pipeline orchestration:** Airflow, Prefect, Dagster, Kubeflow
- **Trigger sources:** Monitoring system alerts, webhooks, cloud events
- **Feature store updates:** Feast, Tecton — trigger retrain when feature data updates

### Human-Triggered Retraining

Sometimes a human needs to intervene.

| Scenario | Who Triggers | Why |
|---|---|---|
| **Recognized failure mode** | ML engineer | Model is systematically wrong on a known segment |
| **Regulatory requirement** | Compliance officer | Model must be revalidated after regulation change |
| **Business event** | Product manager | New product line, pricing model, or user segment |
| **Data quality incident** | Data engineer | Training data was corrupted; retrain on clean data |
| **Edge case discovery** | ML engineer / reviewer | Found an important pattern the model misses |

**Best practice:** Make human-triggered retraining a self-service operation. A button in the monitoring dashboard that says "Retrain on last 30 days of data" — no code changes, no pipeline modifications. This lowers the barrier to fixing the model when something goes wrong.

---

## 7.5 Governance & Ethics

### Bias Detection

Machine learning models can perpetuate or amplify societal biases present in training data.

| Bias Type | Example | Detection Method |
|---|---|---|
| **Demographic bias** | Loan model approves 80% of male applicants, 60% of female applicants with same income | Slice-based metrics by demographic group |
| **Representation bias** | Training data is 95% one ethnicity → model fails on minority groups | Data composition analysis |
| **Label bias** | Historical hiring data reflects past discrimination → model learns it | Audit label distribution by group |
| **Measurement bias** | Proxy metric (e.g., arrest rate for policing) doesn't match true signal | Domain expert review of features and labels |
| **Aggregation bias** | One model for all groups misses subgroup patterns | Segment-level evaluation |

**Bias detection tools:**
| Tool | What It Does |
|---|---|
| **AIF360 (IBM)** | Bias metrics (disparate impact, equal opportunity difference), mitigation algorithms |
| **Fairlearn (Microsoft)** | Fairness dashboard, disparity analysis, mitigation (grid search over fairness constraints) |
| **SHAP / LIME** | Local explanations — identify if certain features (like race or gender proxy) dominate predictions |
| **What-If Tool (Google)** | Interactive fairness analysis, slice comparison |
| **Aequitas** | Bias report cards for model audits |

**Real example:** A hiring model trained on 5 years of historical data learned to downgrade female candidates because past hiring was male-dominated. Slice evaluation revealed 0.92 AUC for male candidates vs 0.68 for female candidates. Fix: balanced training data + fairness constraint (equal opportunity).

**Best practice:** Define protected attributes (race, gender, age, region) before training. Run slice-based evaluation on all protected attributes at every model release. Gate releases on fairness criteria equal to accuracy criteria.

### Auditing Predictions

An audit trail enables investigation of any decision after the fact.

| Audit Requirement | Implementation |
|---|---|
| **Who was predicted?** | Log user_id, request_id, timestamp |
| **What was predicted?** | Log raw score, thresholded decision, confidence |
| **Why was it predicted?** | Log feature values (or SHAP values at prediction time) |
| **Which model?** | Log model version, training date, artifact hash |
| **What happened after?** | Join with outcome (label, actual result) when available |

**Audit log structure:**
```json
{
  "request_id": "uuid",
  "timestamp": "2026-07-18T14:30:00Z",
  "user_id": "12345",
  "model_version": "fraud_v2.3.1",
  "features": {"amount": 299.99, "is_new_user": true, ...},
  "prediction": {"score": 0.87, "decision": "block", "threshold": 0.8},
  "shap_values": {"amount": 0.12, "is_new_user": 0.54, ...},
  "outcome": null
}
```

**Best practice:** Store audit logs in append-only storage (immutable). Retention policy: hot storage for 90 days (frequent queries), cold storage for years (regulatory). Audit logs are not just for compliance — they're the primary debugging tool for production incidents.

**Common stack:**
- **Audit logging:** Immutable log stores (Apache Kafka, AWS CloudTrail, Azure Monitor)
- **Feature logging:** Feature store serving logs, model prediction logs
- **Compliance:** Collibra, Alation (data catalog/governance)

### Responsible ML Practices

| Practice | Why It Matters | Implementation |
|---|---|---|
| **Model cards** | Document model purpose, limitations, evaluation results, bias analysis | Standardized card per model (Google's model cards framework) |
| **Fairness evaluation** | Ensure model doesn't discriminate | Slice metrics per protected group; pre-deployment fairness gate |
| **Interpretability** | Enable debugging and trust | SHAP, LIME, or inherently interpretable model where required |
| **Privacy** | Protect user data | Differential privacy (DP-SGD), anonymization, data minimization |
| **Consent management** | Respect user preferences | "Do not personalize" flag, data deletion pipeline |
| **Human oversight** | Catch edge cases ML can't handle | HITL for decisions above a confidence threshold |
| **Documentation** | Reproducibility and auditability | Training manifests, data lineage, experiment tracking |
| **Regular audits** | Catch regressions early | Quarterly fairness audit + model performance review |

**Model card example sections:**
```
Model ID: fraud_classifier_v2.3.1
Date: 2026-07-18
Task: Binary classification (fraud / not fraud)
Training data: 2.5M transactions from Jan–Dec 2025
Evaluation: 89% precision, 76% recall on Jan 2026 test set
Fairness: AUC within ±0.01 across gender and age segments; +0.03 for new users (known limitation)
Limitations: High false-positive rate for international transactions (25% FP vs 8% domestic)
Intended use: Real-time fraud screening for US-based e-commerce transactions
Out-of-scope: High-value wire transfers, P2P payments, non-US markets
```

**Common stack for governance:**
- **Model registries:** MLflow Model Registry, Hugging Face Hub, SageMaker Model Registry, Azure ML Registry
- **Data catalogs:** Amundsen, DataHub, Atlan
- **Compliance:** Collibra, OneTrust, BigID
- **Bias/fairness:** AIF360, Fairlearn, What-If Tool

**Best practice:** Start with a lightweight model card template (1 page per model). Store it alongside the model artifact in the registry. Require a completed model card as a deploy gate. Expand the template as regulatory requirements evolve.
