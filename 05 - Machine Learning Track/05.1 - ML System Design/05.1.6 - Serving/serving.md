# Model Serving

Model serving is the infrastructure and practice of making model predictions available to users and systems in production. A model that achieves 0.99 AUC in offline evaluation delivers zero business value until it serves predictions at the required latency, throughput, and reliability. Serving is where ML meets software engineering — and where most ML systems fail.

A serving strategy must balance latency, cost, reliability, and accuracy. A real-time fraud model that costs $0.001 per prediction but misses the latency SLA by 50ms will be replaced by a simpler heuristic regardless of its AUC. Every deployment pattern, caching strategy, and fallback mechanism should be designed before the first model goes live — retrofitting reliability after an outage is expensive and erodes trust in ML across the organization.

## 6.1 Serving Architecture

### Batch Prediction Systems

Predictions are computed on a schedule (hourly, daily) and stored for later retrieval.

| Characteristic | Batch |
|---|---|
| **Latency** | Minutes to hours |
| **Throughput** | Very high (millions of rows per run) |
| **Compute** | Distributed (Spark, BigQuery, Trino) |
| **Storage** | Results written to table / key-value store |
| **Cost** | Low per prediction |
| **Use case** | Churn scores → email campaign, daily recommendations, credit scoring |

**Architecture:**
```
Scheduled trigger → Feature pipeline → Model inference → Write predictions to table
                                                              ↓
                                                     Online API reads from table at request time
```

**Best practice:** Store predictions with a primary key (user_id, item_id) and timestamp. The serving API does a simple key lookup — no model inference during the request path.

**Trade-off:** Batch predictions are cheap but always slightly stale. A user who churned 10 minutes ago won't be flagged until tomorrow's batch run.

### Online Inference Systems

Predictions are computed per-request in real-time.

| Characteristic | Online |
|---|---|
| **Latency** | Milliseconds to seconds |
| **Throughput** | Varies (100–100K QPS) |
| **Compute** | Single-node or small cluster (CPU/GPU) |
| **Storage** | Model in memory; features fetched on-demand |
| **Cost** | Higher per prediction |
| **Use case** | Fraud scoring, real-time recommendations, search ranking |

**Architecture:**
```
Request → Feature retrieval (online store) → Model inference → Response
```

**Components:**
- **Serving framework:** TorchServe, TensorFlow Serving, MLflow, BentoML, Triton Inference Server
- **Feature store:** Redis, DynamoDB, Cassandra for low-latency feature lookups
- **Model format:** ONNX, TorchScript, TensorRT for optimized inference

**Best practice:** Keep the critical path minimal — feature retrieval and inference should be < 50ms for latency-sensitive applications. Move non-essential logic (logging, analytics) to async sidecars.

### Streaming Inference

Predictions are computed on a continuous event stream, without explicit request boundaries.

| Characteristic | Streaming |
|---|---|
| **Latency** | Seconds to minutes (event processing time) |
| **Throughput** | Very high (millions of events/second) |
| **Compute** | Stream processor (Flink, Kafka Streams, Spark Streaming) |
| **State** | Stateful — maintains windows, aggregations across events |
| **Use case** | Real-time fraud detection on transaction streams, IoT anomaly detection, real-time personalization |

**Architecture:**
```
Event stream (Kafka) → Stream processor → Feature computation → Model inference → Output stream (predictions)
```

**Best practice:** Use streaming when the decision must be made within seconds of the event, and events arrive continuously. For request-response patterns, use online inference. Batch → streaming → online is a spectrum, not a binary choice — many systems use all three.

---

## 6.2 Latency & Throughput

### Model Size vs Latency

| Model | Size | Inference Latency (CPU) | Latency (GPU) | Notes |
|---|---|---|---|---|
| Logistic regression | KB | 10–50μs | — | Negligible |
| Decision tree (depth 10) | KB | 1–10μs | — | Fastest among ML models |
| Random Forest (100 trees) | MB | 1–10ms | — | Linear in tree count |
| XGBoost / LightGBM | MB–GB | 1–20ms | — | Well-optimized for CPU |
| Small neural network (< 1M params) | MB | 5–50ms | 1–5ms | GPU helps for batch |
| ResNet-50 | ~100MB | 100–500ms | 5–15ms | GPU recommended |
| BERT-base | ~440MB | 500ms–2s | 15–50ms | GPU required for latency-sensitive |
| LLM (7B params) | ~14GB | — | 50–200ms (with optimization) | Needs GPU, quantization |

**Latency budget breakdown (typical 200ms SLA):**
```
Request parsing:       1ms
Feature retrieval:    20ms  (2–3 Redis lookups)
Feature computation:  10ms  (transforms, joins)
Model inference:      40ms  (XGBoost / small NN)
Post-processing:       5ms  (threshold, formatting)
Response:              1ms
Network roundtrip:    50ms  (client → server → client)
Safety margin:        73ms
Total:               200ms
```

**When latency exceeds budget, in order of cost/effort:**
1. Reduce feature count (prune low-importance features)
2. Model quantization (FP32 → FP16 → INT8)
3. Model pruning (remove low-weight connections)
4. Smaller model architecture (distillation, or simpler model class)
5. Hardware upgrade (CPU → GPU, faster instance type)

### Hardware Considerations

| Hardware | Best For | Cost | Latency |
|---|---|---|---|
| **CPU (general purpose)** | Tree-based models, small NNs, batch inference | Low | Moderate |
| **CPU (with AVX-512 / AMX)** | XGBoost, LightGBM (benefits from vectorization) | Low–Medium | Good |
| **GPU (T4, L4)** | Medium NNs, CNNs, BERT-size transformers | Medium | Fast |
| **GPU (A100, H100)** | Large transformers, LLMs, high-throughput | High | Fastest |
| **Inference chips (AWS Inferentia, Google TPU)** | Specific model families at scale | Medium–High | Fast (after warmup) |
| **Edge devices (Jetson, iPhone Neural Engine)** | On-device inference, offline mode | Low (per device) | Fast (no network) |

**Real example:** A content moderation system processing 10M images/day switched from CPU (T2 instances) to GPU (G4dn instances). Inference latency dropped from 800ms to 40ms per image, and total cost decreased 60% because fewer instances were needed despite higher per-instance cost.

### Caching Predictions

Caching avoids redundant computation when the same input is likely to repeat.

| Cache Level | Storage | TTL | Hit Rate | Use Case |
|---|---|---|---|---|
| **L1 — Request-level** | In-memory (request scope) | Request lifetime | Low | Repeated feature lookups in the same request |
| **L2 — Session-level** | In-memory (Redis, Memcached) | Minutes | Medium | User features that rarely change (`user_segment`, `signup_date`) |
| **L3 — Precomputed batch** | Database / KV store | Hours–days | High | Daily scores (churn probability, LTV) |
| **L4 — Result cache** | Redis / CDN | Seconds–minutes | Medium–High | Identical requests in a short window (hot items, trending queries) |

**Cache invalidation:**
- **TTL-based:** Simplest — expire after N seconds. Accept staleness.
- **Event-based:** On user action (purchase, profile update), invalidate that user's cache. Lower staleness but more complex.
- **Write-through:** Precomputed scores are written to cache during batch inference. No explicit invalidation — new batch overwrites.

**Trade-off:** Caching introduces staleness. For fraud detection, a 10-second-old prediction can miss a fraudulent transaction. For content recommendations, 10-minute-old predictions are fine. Match TTL to the rate of change of the prediction target.

**Best practice:** Always cache static features (user signup date, device type). Cache dynamic features only if they change slower than the cache TTL. Never cache predictions for high-stakes decisions where staleness is dangerous.

---

## 6.3 Model Deployment Patterns

### Blue-Green Deployments

| | Blue (Current) | Green (New) |
|---|---|---|
| **Status** | Live traffic | Pre-deployed, idle |
| **Switch** | — | Cutover: all traffic → green |
| **Rollback** | Blue still running, switch back instantly | — |

**Pros:**
- Instant rollback (just switch traffic back)
- No gradual exposure — if green passes pre-deploy checks, it gets full traffic

**Cons:**
- Double infrastructure cost during deployment (both versions running)
- No partial exposure — if green has a subtle bug, all users see it at once
- Needs load balancer or traffic router that supports instant switching

**When to use:** Low-risk models (featured validated thoroughly offline), high-velocity deployments (multiple per day), or when canary infrastructure doesn't exist yet.

### Canary Releases

Gradually shift traffic from old to new model while monitoring metrics. (Detailed in evaluation_strategy.md §5.3.)

**Phases:**
```
Shadow (parallel, no decisions) → 5% → 25% → 50% → 100%
```

**ML-specific considerations:**
- **Traffic splitting:** By user_id hash (ensures a given user sees consistent predictions throughout the canary)
- **Metrics:** Compare primary metric per-traffic-slice. If canary slice's metric drops relative to control, pause or rollback.
- **Durations:** Minimum 1 full business cycle per phase (1 day for daily patterns, 1 week for weekly patterns)

**Real example:** An e-commerce ranking model canary: at 5%, CTR was flat. At 25%, CTR improved 2% but revenue dropped 1% (model optimized for clicks, not purchases). Rolled back, fixed the proxy metric, re-deployed. Without canary, this regression would have hit all users.

### Shadow Models

Run the new model alongside the current one, but discard its predictions for decisions. Log everything for offline analysis.

```
Traffic → Champion model → decision (used)
        → Shadow model → prediction (logged, not used)
                           ↓
                     Compare offline: "What would shadow have decided?"
```

**What shadowing catches:**
- Feature pipeline differences (shadow computed features differently than during training)
- Latency regressions (shadow is slower, but user doesn't feel it — yet)
- Distribution mismatches (shadow's training distribution ≠ production distribution)
- Rare edge cases not in the test set

**Trade-off:** Shadowing doubles inference cost (two models per request) and adds no immediate business value. It's insurance — catch problems before they affect users. Worth it for high-stakes models; skip for low-risk ones.

**Best practice pipeline:**
```
Shadow (N days) → Offline analysis gate → Canary (5% → 25% → 50%) → Full rollout
```

---

## 6.4 System Reliability

### Fallback Strategies

Every serving system needs a fallback chain for when the model is unavailable, slow, or returns garbage.

```
Level 1: ML model prediction (normal path)
Level 2: Cached prediction (if available and TTL hasn't expired)
Level 3: Heuristic / rule-based prediction (simple logic, no ML)
Level 4: Default value (population mean, safe default — e.g., "approve" or "deny")
Level 5: Error response (inform caller that prediction is unavailable)
```

| Fallback Level | Latency | Accuracy | When Triggered |
|---|---|---|---|
| **ML model** | 50ms | Highest | Normal operation |
| **Cached prediction** | 1ms | Good | Model timeout / error, cache hit |
| **Heuristic** | 5ms | Medium | Model down, cache miss |
| **Default value** | 0ms | Low | All upstream systems unavailable |
| **Error response** | 0ms | None | Explicit failure mode — inform consumer |

**Real example — fraud detection fallback:**
```
Normal: ML model (gradient boosted tree) → score → block/approve
If ML times out: use cached 5-minute-old score
If cache miss: rule-based (block if amount > $10K AND new user)
If all fail: default "approve" (minimize friction; log failure for review)
```

### Graceful Degradation

When the model fails, the system should fail **partially**, not **catastrophically**.

| Scenario | Degraded Behavior | User Experience |
|---|---|---|
| Model returns NaN / null | Fall back to cached or heuristic prediction | No visible change |
| Model latency > SLA | Serve cached prediction; trigger alert | Slightly stale prediction |
| Feature store unavailable | Use default feature values (0, mean) | Potentially less relevant results |
| Model artifact corrupted | Route to previous model version | No visible change |
| Entire serving cluster down | Serve from a static precomputed table or default rules | Limited personalization but system stays up |

**Circuit breaker pattern:**
```
Normal → Model errors increase → Circuit opens (stop calling model) → Fallback activated
                                                                        ↓
                                                            After cooldown → Circuit half-open → Test one request → If success, close circuit
```

### What Happens When the Model Is Down?

**Prep before deployment:**

| Question | Answer Needed |
|---|---|
| Can the system survive without ML for 1 hour? | Define fallback behavior |
| How do users experience the outage? | Test degraded mode regularly |
| How long until auto-recovery? | Define cooldown and retry logic |
| Is there a manual override? | Document who can switch to fallback and how |
| What telemetry is exposed? | Dashboard for serving health (error rate, latency, fallback usage) |

**SLIs and SLOs for serving:**

| SLI | Target (SLO) | Measurement |
|---|---|---|
| **p99 latency** | < 200ms for 99.9% of requests | Request-level timing |
| **Error rate** | < 0.1% of requests return 5xx | HTTP status codes |
| **Model freshness** | Predictions use features no older than 60s | Feature timestamp vs current time |
| **Fallback rate** | < 5% of requests use cached/heuristic fallback | Fallback counter |
| **Uptime** | 99.95% (≈ 4 hours downtime/year) | Health check endpoint |

---

## 6.5 Cost Optimization

### Inference Cost Drivers

| Cost Driver | Impact | Optimization |
|---|---|---|
| **Compute (CPU/GPU)** | Largest cost (50–80% of total) | Right-size instance, use batch inference where possible |
| **Model size** | Larger model = more memory = more expensive instance | Quantization, pruning, distillation |
| **Feature retrieval** | Each feature fetch costs latency and I/O | Cache aggressively, reduce feature count |
| **Preprocessing** | Feature computation per request | Precompute where possible |
| **Network egress** | Data transfer costs (especially for large outputs like embeddings) | Compress responses, cache at CDN |
| **Cold starts** | Model loading on first request (minutes for large models) | Prewarm instances, keep-alive |

**Cost per prediction (approximate):**

| Architecture | Cost per 1K predictions | Monthly cost at 100K QPS |
|---|---|---|
| Linear model (CPU, batch) | $0.0001 | $26 |
| XGBoost (CPU, online) | $0.001 | $260 |
| BERT-base (GPU, batch) | $0.01 | $2,600 |
| BERT-base (GPU, online) | $0.05 | $13,000 |
| LLM 7B (GPU, online) | $0.50–$2.00 | $130K–$520K |

**Key insight:** The cost difference between a linear model and an LLM can be 10,000×. The business value must justify the cost.

### Model Compression

| Technique | Size Reduction | Accuracy Impact | Inference Speedup |
|---|---|---|---|
| **Quantization (FP32 → FP16)** | 50% | Negligible | 1.5–2× (GPU) |
| **Quantization (FP32 → INT8)** | 75% | < 1% loss | 2–4× (CPU/GPU) |
| **Pruning (remove 50% of weights)** | 50% | < 1% loss | 1.5–2× (sparse hardware) |
| **Knowledge distillation** | 60–90% | 1–3% loss (student vs teacher) | 3–10× |
| **ONNX Runtime optimization** | — | None | 1.5–3× |
| **TensorRT** | — | None (with FP16/INT8) | 3–8× (GPU) |

**Best practice:** Always apply quantization (FP16 at minimum) before deploying any neural network. It halves memory and doubles throughput with negligible accuracy loss. Profile INT8 if the accuracy drop is acceptable (most models absorb it well).

**Real example:** A BERT-based text classifier was 440MB and took 500ms on CPU. After INT8 quantization + ONNX Runtime optimization, it was 110MB and took 50ms — 10× faster, < 1% accuracy loss. No hardware upgrade needed.

### Traffic Shaping

Control how and when inference requests hit the serving infrastructure.

| Technique | How It Works | Benefit |
|---|---|---|
| **Request batching** | Group multiple requests into one inference call | 5–10× throughput on GPU (amortizes kernel launch overhead) |
| **Prewarming** | Keep model loaded + warm (send dummy requests) | Eliminates cold start latency (minutes → milliseconds) |
| **Adaptive throttling** | Reject or queue requests when latency exceeds threshold | Protects system from overload; maintains p99 SLA |
| **Traffic shifting** | Route non-critical predictions to cheaper/batch infrastructure | Cost savings (batch GPU is cheaper than online GPU) |
| **Regional routing** | Serve from the closest region | Lower latency, lower egress costs |

**Batch sizing example (GPU inference):**

| Batch Size | Throughput (preds/sec) | Latency (p99) | Cost per prediction |
|---|---|---|---|
| 1 | 100 | 10ms | 1.0× (baseline) |
| 8 | 600 | 15ms | 0.17× |
| 32 | 1,800 | 30ms | 0.06× |
| 128 | 4,000 | 100ms | 0.025× |

**Trade-off:** Larger batches = lower cost but higher latency. For online serving, batch up to the max latency budget allows. For batch inference, use the largest batch that fits in GPU memory.

**Best practice for cost:** Batch non-urgent predictions (daily scoring, reporting) on preemptible/spot instances at 60–90% discount. Reserve on-demand or reserved instances for latency-critical online serving.
