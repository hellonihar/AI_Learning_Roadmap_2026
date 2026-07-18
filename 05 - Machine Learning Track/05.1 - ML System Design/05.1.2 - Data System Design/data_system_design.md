# Data System Design

## 2.1 Data Sources

### First-Party vs Third-Party Data

| | First-Party | Third-Party |
|---|---|---|
| **Source** | Collected directly from your users/systems | Bought or licensed from external providers |
| **Control** | Full control over schema, quality, latency | No control; provider can change format or disappear |
| **Cost** | Infrastructure cost only | Often expensive (credit bureau data, demographic enrichments) |
| **Privacy** | Easier to manage consent (direct relationship) | Complex — GDPR, CCPA compliance for sharing data |
| **Example** | User click logs, purchase history | Weather API, economic indicators, firmographic data |
| **Signal strength** | Strong (directly reflects your users) | Weak (correlated, not causal) |

**Best practice:** Start with first-party data. Add third-party only when it measurably improves the model — and have a fallback if the source goes away.

### Logged Data vs User-Generated Data

| | Logged Data | User-Generated Data |
|---|---|---|
| **Generation** | System events, server logs, app telemetry | User actions (ratings, reviews, uploads) |
| **Volume** | High (every request, every click) | Low (most users never rate or review) |
| **Bias** | Reflects real behavior (what users actually did) | Survivorship bias (only active users engage) |
| **Noise** | Low (structured, timestamped) | High (spelling errors, sarcasm, fakery) |
| **Example** | Page views, API latency, add-to-cart events | Star ratings, product reviews, support tickets |

**Key insight:** Logged data is your workhorse for ML — it's abundant, low-bias, and automatically collected. User-generated data is sparse but provides explicit signal.

### Static vs Streaming Data

| | Static (Batch) | Streaming (Real-Time) |
|---|---|---|
| **Arrival** | Daily/hourly dumps | Continuous event stream |
| **Storage** | Data lake, warehouse | Kafka, Kinesis, event bus |
| **Processing** | Spark, SQL, Airflow | Flink, Kafka Streams, Lambda |
| **Latency** | Hours to days | Milliseconds to seconds |
| **Use case** | Training datasets, batch scoring | Real-time features, online learning |

**Typical hybrid architecture:**
- **Batch layer:** Processes historical data → training datasets + daily feature tables
- **Stream layer:** Computes real-time features (e.g., "page views in last 5 minutes") and merges with batch features at inference time

### Training Data vs Serving Data

| | Training Data | Serving Data |
|---|---|---|
| **When** | Historical (weeks/months ago) | Current (now) |
| **Labels** | Available | Not available (must predict) |
| **Features** | Computed offline, can look into past | Computed online, only current context |
| **Volume** | Terabytes (full history) | Per-request (KB) |
| **Quality** | Cleaned, validated, curated | Raw, may have missing values |

**Critical mismatch warning:** If feature computation differs between training and serving, the model sees a different distribution at inference. Common cause: using `avg(price_30d)` in training (looks 30 days back from each row's timestamp) but `avg(price_all_time)` in serving (includes future data).

**Fix:** Use exactly the same feature engineering code in both paths, with consistent time-window semantics.

---

## 2.2 Data Collection & Logging

### What to Log (Features, Predictions, Outcomes)

Log everything needed to debug, improve, and audit the system:

```
Log entry:
  - request_id (traceability key)
  - timestamp (when prediction was made)
  - input_features (all feature values used)
  - model_version / model_id
  - prediction (raw score, probability, class)
  - confidence (if applicable)
  - latency (feature computation + inference time)
  - outcome (if/when it arrives — ground truth label)
  - metadata: user_id, session_id, experiment_id (A/B bucket)
```

**Storage rule of thumb:** Keep raw logs for 30 days (hot), then compress to feature aggregates for longer retention (cold). Raw logs grow fast — at 10K QPS, a year of logs is ~300 TB.

### Schema Design for ML Logs

| Field | Type | Example | Notes |
|---|---|---|---|
| `request_id` | UUID | `a1b2c3d4-...` | Join key for features → prediction → outcome |
| `timestamp` | TIMESTAMP | `2026-07-18T14:30:00Z` | UTC, always |
| `features` | JSON / Struct | `{"price": 29.99, "click_7d": 42}` | Sparse OK; log only features used |
| `prediction` | FLOAT | `0.87` | Raw score, not thresholded |
| `model_version` | STRING | `v2.3.1` | Maps to exact trained artifact |
| `outcome` | FLOAT | `1.0` (if label arrives) | May be delayed or null |
| `feature_compute_ms` | INT | `12` | Debug latency regressions |

**Schema evolution:** Use a schema registry (Avro, Protobuf) so old and new log formats coexist during deployments.

### Handling Delayed and Missing Labels

- **Delayed labels** (e.g., loan default takes 12 months): Store prediction + features, and when label arrives, join by `request_id`. Train only on rows with observed outcomes.
- **Missing labels** (e.g., user never returned): Could mean negative outcome (churn) or censored data (user is still active). Decide per problem — and document the assumption.
- **Partial feedback:** Only a fraction of predictions receive labels (e.g., only reviewed fraud cases get a "real" label). The labeled subset is biased toward ambiguous cases.
  - **Mitigation:** Log the sampling mechanism so you can correct for selection bias during evaluation.

### Avoiding Data Leakage at Collection Time

| Leakage Type | Example | Prevention |
|---|---|---|
| **Temporal leakage** | Using `avg(price_30d)` that includes future prices | Rigorous time-based partitioning; never compute window functions across partition boundaries |
| **Feature leakage** | Using `is_purchase` as a feature to predict `is_purchase` | Audit feature → label relationship; use feature importance to detect suspiciously high-weight features |
| **Group leakage** | Same user in train and test splits | Split by user_id, not by event |
| **Label leakage** | Using label-derived features (e.g., "number of future purchases") | Only compute features from data available **before** the prediction timestamp |

**Golden rule:** At collection time, ask: "If I were making this prediction in production right now, would this feature value be available?" If no, don't log it as a feature.

---

## 2.3 Data Storage & Versioning

### Data Lakes vs Data Warehouses

| | Data Lake | Data Warehouse |
|---|---|---|
| **Data type** | Raw, unstructured, any format | Structured, cleaned, schema-on-write |
| **Purpose** | Exploration, ML training, data science | BI reports, dashboards, analytics |
| **Storage** | Cheap (object storage — S3, ADLS, GCS) | Expensive (columnar — Redshift, BigQuery, Snowflake) |
| **Schema** | Schema-on-read (flexible) | Schema-on-write (rigid) |
| **ML use** | Store raw logs, train on full history | Serve clean label tables, feature tables for training |

**Common ML architecture:**
```
Raw logs → Data Lake (S3/ADLS)
  → Feature pipeline → Feature Store (for training + serving)
  → Label pipeline → Label table (in warehouse)
  → Training dataset = Feature Store JOIN Label table
```

### Immutable Datasets

**Rule:** Never modify a dataset in place. Write a new version instead.

- **Why:** Reproducibility. If you overwrite `train_v1.parquet`, you can never recreate the model exactly.
- **How:** Partition by date and model version (`data/train/{model_version}/{date}/`). Append-only.
- **Cost:** Storage is cheap. Debugging irreproducible models is expensive.

### Dataset Versioning Strategies

| Strategy | How It Works | Best For |
|---|---|---|
| **Timestamp partitions** | `data/train/dt=2026-07-18/` | Daily retraining, simple |
| **Git-LFS / DVC** | Git commit = dataset version hash | Small teams, experiment tracking |
| **LakeFS / Nessie** | Git-like branches for data lakes | Multiple concurrent experiments |
| **Feature Store versioning** | Feast, Tecton — versioned feature views | Production ML with reuse across teams |

**Minimum viable:** Store dataset creation date and a hash of the preprocessing config. If params change, the hash changes → new dataset.

### Reproducibility Requirements

To reproduce any past model exactly, you need:

1. **Code version** — Git commit hash of training pipeline
2. **Dataset version** — Path + file listing (or DVC hash)
3. **Feature config** — Which features, how computed, window sizes
4. **Hyperparameters** — All training params (including random seed)
5. **Environment** — Docker image or `requirements.txt` + Python version
6. **Label definition** — How labels were generated (e.g., "purchase within 7 days")

**Store all six in a single manifest file alongside the trained model artifact.**

---

## 2.4 Data Quality & Validation

### Missing Values, Outliers, Duplicates

| Issue | Detection | Impact | Fix |
|---|---|---|---|
| **Missing values** | `isnull().sum()` per feature | Model may silently degrade if not handled | Impute (mean, median, model-based) or use models that handle NaN (XGBoost) |
| **Outliers** | IQR, Z-score, DBSCAN | Skew loss, destabilize training | Clip, transform (log), or remove if clearly erroneous |
| **Duplicates** | Hash-based dedup, groupby count | Train-test contamination (same row in both splits) | Dedup by `request_id` or `(user_id, timestamp)` |
| **Constant features** | `nunique() == 1` | Wasted compute, no signal | Drop before training |
| **ID columns** | High cardinality, no predictive value | Accidental leakage (model memorizes IDs) | Filter out from feature set |

**Build a data quality report** that runs before each training job and alerts on regressions.

### Data Validation Checks

**Runtime checks (before training):**

| Check | Example | Action |
|---|---|---|
| **Schema validation** | Feature `price` changed from FLOAT to STRING | Fail pipeline |
| **Range check** | `age` should be 0–120 | Fail or clip |
| **Null rate** | `click_7d` is 95% null (was 5% yesterday) | Alert, investigate source |
| **Cardinality** | `city` suddenly has 10K unique values (was 500) | Likely data corruption or ID leaking |

**Use tools:** Great Expectations, Pandera, Deequ — define expectations as code, versioned alongside the pipeline.

### Distribution Monitoring

Monitor how feature distributions change over time:

| Monitor | What It Catches | Alert When |
|---|---|---|
| **Feature mean / std** | Feature drift | > 2σ from baseline |
| **PSI (Population Stability Index)** | Distribution shift | PSI > 0.1 |
| **KS test / JS divergence** | Statistical distribution change | p-value < 0.01 |
| **Null rate per feature** | Data source degradation | > 2x baseline |

**UI example:** Dashboard with per-feature distribution plots, color-coded green/yellow/red based on drift severity.

### Silent Data Corruption

The most dangerous data quality issue — no errors, no warnings, just wrong values.

| Scenario | Example | Detection |
|---|---|---|
| **Off-by-one** | Logs written to wrong date partition | Compare row counts day-over-day |
| **Encoding corruption** | UTF-8 string arrives as Mojibake | Regex check on string patterns |
| **Unit mismatch** | Price in cents vs dollars | Cross-validation with expected range |
| **Schema drift** | API adds a field, column order shifts | Schema registry + explicit column mapping |
| **Timestamp skew** | Server clock drifts, events misordered | NTP monitoring, check `max(timestamp) - now()` |

**Defense:** Checksums on raw data, row count alerts, and a "known good" baseline dataset that every new pipeline run is compared against.

---

## 2.5 Data Drift

### Covariate Drift vs Label Drift vs Concept Drift

| Type | What Changes | Example | Impact |
|---|---|---|---|
| **Covariate drift** | Input feature distribution `P(X)` | Users get 30% older over 2 years | Model sees unfamiliar inputs → predictions less reliable |
| **Label drift** | Label distribution `P(Y)` | Fraud rate drops from 2% → 0.5% | Baseline shifts; evaluation against old threshold is misleading |
| **Concept drift** | Relationship `P(Y|X)` changes | Same shopping behavior now signals different purchase intent (post-COVID) | Model is wrong even on inputs it's confident about — most dangerous |

**Visual distinction:**
- Covariate drift: input distribution shifts, but the mapping stays the same
- Concept drift: input distribution might be identical, but the label mapping changed

### When Drift Matters and When It Doesn't

**It matters when:**
- The model is used for high-stakes decisions (loans, medical, fraud)
- The deployment is long-lived (weeks/months without retraining)
- The environment is known to change seasonally (holiday shopping, tax season)
- The user base or product changes (new feature launch, new geography)

**It doesn't matter when:**
- The model is retrained daily on fresh data (drift is captured automatically)
- The input space is constrained and controlled (e.g., sensor readings in a factory)
- The relationship is stable (physics-based models)
- The cost of a false alert exceeds the cost of degraded model performance

**Real example:** A recency-frequency-monetary model for customer scoring might not need drift detection if it's retrained every night — drift just means the model adapts. But a fraud model that requires regulatory approval might need drift monitoring and documented retraining decisions.

### Drift Detection Strategies

| Strategy | How It Works | Pros | Cons |
|---|---|---|---|
| **Fixed window comparison** | Compare last N days vs baseline (PSI, KS test) | Simple, interpretable | Window size matters; slow to detect gradual drift |
| **EWMA (Exponentially Weighted Moving Average)** | Weight recent observations more heavily | Fast detection of gradual drift | More parameters to tune |
| **Online detectors** | ADWIN, CUSUM — adapt to stream | No fixed window needed; constant memory | Harder to interpret |
| **Error-based detection** | Monitor accuracy on a held-out fresh sample (if labels arrive quickly) | Direct signal of model degradation | Requires labels; delayed |

**Tiered alerting approach:**

| Level | Condition | Action |
|---|---|---|
| **Info** | PSI > 0.05, < 0.1 | Log, no action |
| **Warning** | PSI > 0.1, < 0.2 | Page ML team, log distributions |
| **Critical** | PSI > 0.2, or any concept drift detected | Auto-disable model → fallback to rules or previous model version |

**Important:** Drift detection should trigger investigation, not panic. Many drifts are benign (seasonal, product changes). Invest in root cause analysis tooling before over-alerting.
