# Feature System Design

## 3.1 Feature Definition

### Static vs Dynamic Features

| | Static | Dynamic |
|---|---|---|
| **Changes** | Rarely or never (e.g., user signup date) | Frequently (e.g., session duration, click counts) |
| **Storage** | Single write, many reads | Updated continuously or computed on-the-fly |
| **Computation cost** | Cheap (compute once) | Varies — can be expensive for real-time aggregates |
| **Example** | `user_country`, `account_created_date`, `device_type` | `page_views_last_5min`, `cart_total`, `avg_session_time_7d` |
| **Freshness concern** | None (changes slowly or not at all) | High — stale dynamic features degrade predictions |

**Best practice:** Cache static features aggressively (they never invalidate). For dynamic features, clearly define **freshness SLA** (e.g., "computed from events up to 5 seconds ago").

### Real-Time vs Batch Features

| | Batch Features | Real-Time Features |
|---|---|---|
| **Computation** | Scheduled (hourly/daily) — Spark, SQL | On-demand or micro-batch — Flink, Kafka Streams |
| **Latency** | Minutes to hours | Milliseconds to seconds |
| **Consistency** | High (same value for all requests in a window) | Eventually consistent (duplicate events, ordering) |
| **Storage** | Data lake / warehouse | KV store (Redis, DynamoDB) or in-memory |
| **Cost** | Cheap per-row | Expensive per-row |
| **Use case** | `avg_purchase_30d`, `user_segment` | `requests_last_minute`, `current_session_page` |

**Real-world split:** Batch features for stable, historical patterns (user lifetime value, average behavior). Real-time features for urgency signals (recent activity burst, session context).

### Aggregations, Windows, Freshness

**Common aggregation types:**
- **Count:** events in a window (`click_count_7d`)
- **Sum / avg / std:** continuous values (`avg_order_amount_30d`)
- **Min / max:** extremes (`max_purchase_ever`)
- **Rate / ratio:** normalized (`clicks_per_session`)
- **Rank / percentile:** relative position (`spend_percentile`)

**Window types:**

| Window | Example | Freshness | Storage |
|---|---|---|---|
| **Fixed tumbling** | `click_count_last_24h` | Updated every 24h | Cheap (precomputed daily) |
| **Sliding** | `click_count_last_7d` from today | Updated daily (window slides) | Cheap but stale at window edges |
| **Cumulative** | `total_purchases_all_time` | Never resets | Cheap, but older data dominates |
| **Real-time sliding** | `clicks_last_5min` (by event timestamp) | ~seconds | Expensive (must process stream) |

**Freshness requirement depends on the prediction:**
- Churn model: 1-day-old features are fine
- Fraud detection: 1-minute-old features can miss an attack
- Ad ranking: 1-second-old features can lose revenue

**Rule:** Match feature freshness to the rate at which the underlying signal changes.

---

## 3.2 Feature Engineering as a System

### Feature Reuse Across Models

Building features once and reusing them across models reduces duplication, computation cost, and maintenance burden.

**How to enable reuse:**
- **Feature registry:** Central catalog (name, owner, computation logic, freshness, dependencies, schema). Feast, Tecton, or a simple YAML file.
- **Feature library:** Shared Python package for feature computation, tested and versioned.
- **Conventions:** Naming (`user_avg_purchase_30d`), units (always USD cents), null handling (always -1, never 0).

**Reuse levels:**

| Level | Description | Complexity |
|---|---|---|
| **Ad-hoc** | Each team computes its own features | Simple to start, impossible to maintain |
| **Shared code** | Common feature functions in shared library | Medium |
| **Feature store** | Central computation + serving infrastructure | High — worth it at 5+ models, 50+ features |

### Feature Coupling Problems

**Coupling** happens when changing one feature inadvertently affects others.

| Coupling Type | Example | Impact |
|---|---|---|
| **Data coupling** | Two features share the same raw table; a schema change breaks both | Feature A breaks even if it didn't change |
| **Logic coupling** | Feature A = `log(avg_purchase)`, Feature B = `avg(purchase_log)` — different semantics, similar names | Confusion, bugs when maintaining |
| **Temporal coupling** | Feature A uses `click_count_7d`, Feature B uses `click_count_7d` from a different time zone | Silent misalignment |
| **Ownership coupling** | Feature A owned by Team X, Feature B depends on A's output | Feature B breaks when Team X changes A without notice |

**Mitigation:** Feature registry with ownership, dependency graphs, schema tests, and breaking-change notifications.

### Feature Ownership and Governance

| Role | Responsibility |
|---|---|
| **Feature owner** | Defines the feature, keeps it documented, responds to breakages |
| **Consumer** | Uses the feature; responsible for understanding its semantics |
| **Platform team** | Maintains feature computation infrastructure, monitors freshness |

**Governance practices:**
- **Feature lifecycle:** Proposed → In development → Production → Deprecated → Removed
- **Change notification:** Any change to a production feature must notify all consumers (via slack, email, or registry webhook)
- **Deprecation window:** Mark deprecated features 30 days before removal; break nothing silently
- **Access control:** Some features contain PII or proprietary signals — restrict by team

---

## 3.3 Training–Serving Skew

### Causes of Skew

| Cause | Training | Serving | Impact |
|---|---|---|---|
| **Feature code mismatch** | Python pandas (offline) | Java (online) | Different values for same logic |
| **Data source mismatch** | Hive table (daily snapshot) | Kafka stream (latest events) | Different distributions |
| **Time semantics** | Window computed with future data | Window computed with past data | Inflated accuracy in training |
| **Missing value handling** | Drop rows with NaN | Model receives NaN → 0.0 default | Silent degradation |
| **Joins** | Inner join (loses users without purchases) | Left join (user present with null features) | Distribution shift |
| **Feature normalization** | Fit scaler on full training set | Apply scaler from artifact (correct, but easy to skip) | Out-of-range values |

### Feature Computation Mismatches

**The root cause:** Different code paths for training and serving.

**Training path:**
```
Raw events → Spark/ETL → Feature DataFrame → Train model
```

**Serving path:**
```
Raw event → REST API → Feature computation → Model prediction
```

If the two paths differ even slightly (join logic, aggregation window, null handling), the model sees different inputs.

**Fix — "Same Code" Rule:**
- Write feature computation **once** as a pure function
- Use it in both training and serving
- If the serving environment can't run the same code (e.g., Python for training, Java for serving), transpile or wrap in a microservice

### Time-Travel Correctness

**The problem:** At training time, you have all data — including events that happened *after* the prediction timestamp. A naive feature computation will accidentally use future data.

**Wrong:**
```python
# Computes avg over ALL user purchases (including future)
df['avg_purchase'] = df.groupby('user_id')['amount'].transform('mean')
```

**Correct (time-travel):**
```python
# For each prediction row, compute avg only from past purchases
df['avg_purchase'] = df.groupby('user_id').apply(
    lambda g: g.sort_values('purchase_time')['amount']
                .expanding().mean().shift(1)
)
```

**In practice:** Use a feature store that handles point-in-time joins automatically, or write explicit time-travel SQL queries with `WHERE event_timestamp < prediction_timestamp`.

### Preventing Skew by Design

| Principle | Implementation |
|---|---|
| **Log features at prediction time** | Store the exact feature vector used for each prediction alongside the prediction ID. For debugging skew, compare logged features against recomputed training features. |
| **Feature store with point-in-time** | Feast, Tecton, or custom system that retrieves feature values as of a given timestamp — same logic for training and serving. |
| **Single code path** | Shared feature library used in both batch pipelines and online serving. |
| **Monitor feature distributions** | Compare logged serving features vs training features. PSI > 0.1 triggers investigation. |
| **Shadow inference** | Run serving features through the same pipeline that generated training features and compare outputs before committing. |

**Golden rule:** If you can't run the exact same feature code on a historical record as you do on a live request, you have skew.

---

## 3.4 Feature Stores

### Why Feature Stores Exist

Feature stores solve three problems that emerge when multiple models share features:

1. **Inconsistency:** Without a feature store, the same feature is computed differently in training vs serving (skew) and across teams (different names, units, semantics).
2. **Duplication:** Five teams independently compute `avg_purchase_30d` — five pipelines, five copies of the same logic, five debugging sessions when it breaks.
3. **Time-travel:** Without point-in-time join support, each team reinvents temporal correctness, often incorrectly.

### Online vs Offline Feature Stores

| | Offline | Online |
|---|---|---|
| **Purpose** | Training dataset generation | Low-latency inference |
| **Storage** | Parquet (S3, ADLS, GCS) | KV store (Redis, DynamoDB, Cassandra) |
| **Serving** | Batch queries (Spark, SQL) | Single-key lookups (GET user:123) |
| **Latency** | Minutes to hours | Milliseconds |
| **Point-in-time** | Yes (core feature) | No (serves latest value) |
| **Throughput** | High (full table scans) | High (key-value lookups) |
| **Example** | Feast offline store, Tecton offline | Feast online store, Tecton online |

**Common pattern:**
```
Batch pipeline → writes to offline (Parquet) + online (Redis) simultaneously
Training: query offline store with point-in-time
Serving: query online store by entity key
```

### When Not to Use a Feature Store

- **You have 1–2 models** and < 20 features — a shared Python module + config is simpler
- **Your features are simple** — a few SQL aggregations, no time-travel needed
- **Your team is small** — the operational overhead of running Feast/Tecton isn't justified
- **Your infrastructure is nascent** — add a feature store after you have basic CI/CD, monitoring, and deployment working

**Anti-pattern:** Adding a feature store to fix a team culture problem (no communication, no standards). It won't — fix the culture first, then add tooling.

**Migration path:**
```
No store → Shared Python package → Versioned feature table (Parquet) → Feature store
```

---

## 3.5 Feature Freshness & Cost

### Freshness vs Latency Tradeoffs

| Freshness | Computation | Latency | Cost | Best For |
|---|---|---|---|---|
| **Real-time** (< 1s) | Stream processing (Flink, Kafka) | On every request | High | Fraud, recommendations, ads |
| **Near real-time** (1–60s) | Micro-batch (every 10s) | Per micro-batch | Medium | Personalization, feed ranking |
| **Batch (hourly)** | Scheduled job (Airflow) | 1h stale | Low | Churn prediction, reporting |
| **Batch (daily)** | Nightly ETL | 24h stale | Very low | User segmentation, lifetime value |

**Key insight:** Pay for real-time freshness only where a stale feature would cause a bad prediction. The difference between a 5-second-old `click_count` and a 10-minute-old one is often negligible for non-fraud models.

### Expensive Features vs Cheap Proxies

| Expensive Feature | Cost Driver | Cheap Proxy | When Proxy Fails |
|---|---|---|---|
| `user_sentiment_score` (NLP model) | Full transformer inference | `n_positive_keywords` | Sarcasm, context matters |
| `purchase_intent_llm` (LLM call) | $$$ per prediction | `add_to_cart + page_depth` | New user, no behavior |
| `credit_risk_gradient` (GBM with 500 features) | Complex inference + many dependencies | `credit_score + income_bracket` | Thin-file users (no credit history) |
| `real_time_location_precision` (GPS + inference) | Compute on device or server | `city_level_location` (from IP) | Delivery ETA needs precise location |
| `product_embedding_similarity` | Embedding lookup + nearest-neighbor search | `category_match + brand_match` | Cross-category recommendations |

**Decision rule:** Use the expensive feature if the improvement in prediction quality justifies the additional latency and cost. A/B test to measure the gap.

### Caching Strategies

| Cache Tier | Scope | TTL | Hit Rate | Use Case |
|---|---|---|---|---|
| **L1 (in-memory, local)** | Current request / session | Seconds | Low | Session context, repeated lookups |
| **L2 (in-memory, distributed)** | Redis / Memcached | Minutes | Medium | User features that change slowly (`user_segment`) |
| **L3 (precomputed batch)** | Parquet / database | Hours–days | High | Stable features (`avg_purchase_30d`) |
| **L4 (CDN / edge)** | Geographic region | Minutes | Medium | Location-based features, weather |

**P95 rule:** Features that don't change within a request should be cached for the request's lifetime. Features that change slowly (minutes/hours) should be precomputed, not computed per-request.

**Cache invalidation:**
- **TTL-based:** Simplest — expire after N seconds. Accept staleness.
- **Event-based:** On user action (purchase, signup), invalidate user's feature cache. More complex but lower staleness.
- **Write-through:** Feature computation pipeline writes to both offline and online store simultaneously. No explicit invalidation needed (latest value always available).
