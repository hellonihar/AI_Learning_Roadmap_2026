# System Design & Scalability — Q&A

## Q1: Design a real-time fraud detection system handling 10K QPS.

**1. Clarify requirements:**
- Latency: <100ms (blocking decision)
- Accuracy: minimize false positives (annoy users) and false negatives (lose money)
- Data: transaction features (amount, location, merchant, device, user history)
- Scale: 10K QPS peak, growing 2x/year

**2. High-level design:**
```
Transaction → Load Balancer → Feature Store → Model Ensemble → Decision → Response
                                 ↑                      ↓
                           Stream Processor        Human Review Queue
                                ↑
                         Event Bus (Kafka)
```

**3. Components:**
- **Feature store** (Feast/Tecton) — real-time features (velocity, geo-anomaly) + batch features (avg spend)
- **Model ensemble** — lightweight model (XGBoost/LightGBM, <5ms) for fast decisions + periodic deep model (Transformer, ~50ms) for complex patterns
- **Shadow mode** — new models run in shadow before deployment; compare decisions offline
- **Human-in-loop** — flag uncertain predictions (model confidence < threshold) for manual review

**4. Scaling:**
- Horizontal scaling of inference servers behind load balancer
- Redis-based feature cache for hot users
- Shard by user_id for consistent feature lookups

**5. Trade-offs:**
- High recall → more false positives → user friction
- Deep model → better accuracy → higher latency → need async fallback

---

## Q2: How would you serve a 70B parameter model with <100ms latency?

**Constraints:** 70B params, <100ms per token (or for prefill), cost-efficient.

**Optimizations:**

1. **Quantization** — FP16 → INT4 (35GB → 17.5GB → fits on 1-2 GPUs)
2. **Tensor parallelism** — shard across 2-4 GPUs. With 4x H100, each handles ~17.5B params
3. **Continuous batching** — vLLM/TensorRT-LLM to maximize GPU utilization
4. **KV cache optimization** — GQA (most 70B models already use it), prefix caching, kv cache quantization
5. **Speculative decoding** — 7B draft model verifies quickly, target model generates 3x fewer tokens
6. **Flash Attention 2/3** — efficient attention kernels, especially for long contexts
7. **Static batching** — batch similar-length requests together (reduces padding waste)

**Architecture:**
```
LB → GPU Node (4x H100, TP=4) → vLLM w/ continuous batching
         ↕
   KV Cache (prefix cache in host memory + GPU)
```

**Expected performance:** ~70B model with 4x H100, continuous batching, INT4: ~50-80ms TTFT, ~30-50ms/token.

---

## Q3: Design a feature store for a large-scale recommendation system.

**Requirements:** Low-latency online lookups (<10ms), high-throughput batch serving for training, point-in-time correctness.

**Architecture:**

```
                    ┌──────────────────┐
                    │  Feature Registry │
                    └────────┬─────────┘
                             │
    Batch Sources ──► Batch Pipeline ──► Online Store (Redis/DynamoDB)
                             │
    Stream Sources ─► Stream Pipeline ──► Offline Store (Parquet/S3)
                             │
                    ┌────────┴─────────┐
                    │ Point-in-time Join│
                    └──────────────────┘
```

**Key features:**
- **Online store:** Redis/Aerospike for <5ms lookups. Key = (entity_id, feature_name). TTL-based expiry.
- **Offline store:** Parquet on S3/ADLS, partitioned by date. Used for training data generation.
- **Feature registry:** metadata catalog (feature name, type, source, owner, freshness SLA)
- **Point-in-time joins:** critical for avoiding data leakage — join features as they existed at the time of the label

**Trade-offs:**
- Consistency vs. latency — eventually consistent for online, strongly consistent for offline
- Freshness vs. cost — real-time features cost more to compute and store

---

## Q4: What breaks when your training data grows from 1TB to 10PB?

**Issues at different scales:**

1. **Storage:** 1TB fits on a local SSD. 10PB needs object store (S3/ADLS) with partitioning
2. **I/O:** Single-process data loading fails. Need data loading frameworks (Petastorm, NVIDIA DALI, Apache Arrow + Parquet)
3. **Preprocessing:** Must move from pandas to distributed processing (Spark, Ray, Dask)
4. **Training:** Single GPU training impossible. Need distributed training: data parallel → model parallel → FSDP
5. **Validation:** Can't run full evaluation on 10PB — need stratified sampling, approximate metrics
6. **Versioning:** git-lfs breaks. Need DVC/Delta Lake with manifest-based versioning
7. **Cost:** Data transfer costs dominate. Co-locate compute with storage (same region, same AZ)

**Recommended architecture at 10PB scale:**
- Data in Delta Lake (parquet, z-order clustered, partitioned by date)
- Preprocessing with Spark/Ray
- Streaming data loading (avoid loading entire dataset; use tf.data/IterableDataset with sharding)
- Distributed training with FSDP + gradient checkpointing
- Evaluation on sampled subsets with confidence intervals

---

## Q5: Design a multi-region ML deployment architecture.

**Goal:** Low latency globally, disaster recovery, consistent behavior.

**Architecture:**

```
Region A (US-East)         Region B (EU-West)        Region C (AP-Southeast)
  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
  │ Inference Pod │          │ Inference Pod │          │ Inference Pod │
  └──────┬───────┘          └──────┬───────┘          └──────┬───────┘
         │                         │                         │
  ┌──────┴───────┐          ┌──────┴───────┐          ┌──────┴───────┐
  │ Feature Cache│          │ Feature Cache│          │ Feature Cache│
  └──────┬───────┘          └──────┬───────┘          └──────┬───────┘
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
                         ┌────────┴────────┐
                         │  Global Model   │
                         │  Registry (S3)  │
                         └─────────────────┘
```

**Key decisions:**
- **Model distribution:** push model artifacts to each region's S3, warm up on deploy
- **Data residency:** EU traffic stays in EU (GDPR). Train on region-specific data if needed
- **Global vs. regional models:** global model for consistent behavior, regional fine-tuning for local patterns
- **Failover:** Route traffic to nearest healthy region. DNS-based + application-level health checks
- **Feature serving:** local feature cache with async replication. Accept eventual consistency
- **Monitoring:** aggregate metrics globally but alert per-region

---

## Q6: Design an end-to-end ML pipeline for a content moderation system.

**Pipeline stages:**

1. **Ingestion** — Kafka topic for uploaded content. Schema: content_id, type (text/image/video), timestamp
2. **Feature extraction** — text embeddings, image hash (for known CSAM), video keyframe extraction
3. **Model inference:**
   - Tier 1 (fast, cheap): blocklist + small classifier (<10ms) — catches 80%
   - Tier 2 (accurate): deep model (ViT + LLM, ~500ms) — catches 15%
   - Tier 3 (human review): uncertain predictions routed to moderators
4. **Action** — allow, flag for review, or block. Write decision to audit log
5. **Feedback loop** — moderator decisions → labeled data → retraining

**Scalability considerations:**
- Async pipeline for non-urgent content (profile photos)
- Sync pipeline for user-reported content (need fast decision)
- Batching for GPU efficiency (process 64 images at once)
- Cold start: new content types need initial dataset and human review

---

## Q7: How would you architect a model training infrastructure for 1000+ GPUs?

**Compute layer:**
- GPU nodes: 8x H100 per node, NVLink-connected within node, InfiniBand between nodes
- Orchestration: Kubernetes + KubeFlow / Slurm / Ray
- Job scheduling: priority queues, gang scheduling (all-or-nothing GPU allocation)

**Training framework:**
- FSDP (Fully Sharded Data Parallel) for parameter sharding across GPUs
- Activation checkpointing to reduce memory (trade compute for memory)
- Mixed precision (BF16) with FP32 master weights
- Gradient accumulation for large effective batch sizes

**Storage:**
- High-throughput parallel filesystem (Lustre/GPUDirect Storage) for checkpoints
- Object store (S3) for datasets, model artifacts
- Local NVMe for data caching

**Reliability:**
- Automatic checkpointing every N steps
- Preemption handling: save state, migrate to available nodes
- Health monitoring: NCCL timeout detection, ECC error monitoring
- Job queuing with priority and preemption policies

**Key bottleneck:** Network communication for gradient sync. Use hierarchical all-reduce for efficiency.

---

## Q8: Design a real-time personalization system.

**Data flow:**
```
User Event → Kafka → Stream Processor → Feature Store → Model Server → Response
                         ↕                              ↓
                   Batch Processor               Experimentation Framework
                         ↓                              ↓
                   Offline Store ←─────── A/B Results ──────
```

**Components:**
- **Real-time feature computation** — user session features, last-clicked item, search queries
- **Batch feature computation** — user embeddings, item embeddings, cohort statistics (computed daily)
- **Model** — two-tower DNN (user tower + item tower), ANN search for candidate generation, then ranking model
- **Experimentation** — A/B framework with statistical significance testing
- **Cold start** — content-based recommendations for new users/items, bandit algorithm for exploration

**Latency budget:** 200ms total: 50ms for feature retrieval, 50ms for candidate generation, 50ms for ranking, 50ms for post-processing.

---

## Q9: Compare batch vs. streaming inference. When would you use each?

| Aspect | Batch Inference | Streaming Inference |
|--------|----------------|-------------------|
| Latency | Minutes to hours | Milliseconds to seconds |
| Trigger | Scheduled (daily/hourly) | Event-driven (real-time) |
| Cost | Lower (efficient batching) | Higher (idle capacity) |
| Infrastructure | Spark, Airflow | Kafka + stream processor |
| Use case | Daily recommendations, reports | Fraud detection, chatbots |
| Model update | Periodic (daily) | Real-time (bandit learning) |

**Hybrid pattern:** Use batch for base predictions (e.g., "your daily recommendations"), streaming for real-time signals (e.g., "since you just clicked this...").

---

## Q10: Design a data pipeline that handles both structured and unstructured data at petabyte scale.

**Architecture:**
1. **Landing zone** — raw data lands in object store (S3/ADLS/GCS), partitioned by source and timestamp
2. **Catalog** — Data Catalog with schema inference for structured, metadata extraction for unstructured
3. **Batch pipeline** — Spark/Databricks for structured ETL, distributed processing for unstructured
4. **Stream pipeline** — Kafka → stream processor for real-time structured events
5. **Processing:**
   - Structured: Spark SQL, dbt transformations, quality checks (Great Expectations)
   - Unstructured: Ray for video processing, Spark NLP for text, distributed image processing
6. **Storage** — Delta Lake/Iceberg for structured, object store with manifest files for unstructured
7. **Serving** — structured to feature store/warehouse, unstructured features to vector DB

**Key challenge:** Schema evolution for structured, content understanding for unstructured. Use schema-on-read for flexibility.

---

## Q11: How do you handle cold start problems in recommendation systems?

**For new users:**
- Onboarding questionnaire (if applicable)
- Content-based recommendations (popular items in their region/demographic)
- Explore-then-exploit: Thompson sampling or epsilon-greedy
- Transfer from similar users (if any signals exist)

**For new items:**
- Content-based features (metadata, image embeddings, description)
- Create a "new items" booster with exploration traffic
- Use popularity as a prior, decay confidence as item ages with low interaction

**For new models:**
- Shadow deployment — run in parallel with existing model, compare offline
- Bandit approach — allocate small traffic % to new model, increase as confidence grows
- Historical simulation — backtest on logged data with importance sampling

---

## Q12: Design a system for A/B testing ML models at scale.

**Components:**
1. **Experiment framework** — assign users to control/treatment consistently (consistent hash on user_id)
2. **Traffic splitting** — 90/10, 80/20 split, dynamically adjustable
3. **Model serving** — route requests based on experiment assignment
4. **Metrics pipeline** — real-time metrics (latency, CTR) + daily batch metrics (revenue, retention)
5. **Statistical engine** — frequentist (t-test) or Bayesian (probability of being best) with sequential testing
6. **Automation** — if treatment wins with 95% confidence → auto-promote; if loses → auto-rollback

**Key considerations:**
- **Interference** — user A in treatment affects user B in control (e.g., social features). Use network A/B or cluster-based randomization
- **Multiple comparisons** — correct for many metrics or many variants (Bonferroni, Benjamini-Hochberg)
- **Early stopping** — sequential testing (always valid p-values) avoids peeking bias
- **Long-term effects** — short-term metric improvement may not predict long-term retention. Run holdout groups

---

## External Resources

- [AppScale: 15 AI System Design Questions + Diagrams (2026)](https://appscale.blog/en/blog/ai-system-design-interview-top-15-questions-architecture-diagrams-2026)
- [System Design Handbook: ML System Design Guide (2026)](https://www.systemdesignhandbook.com/guides/ml-system-design/)
- [Fonzi AI: System Design Interview Prep 2026 (ML & GenAI)](https://fonzi.ai/blog/system-design-interview)
- [GitHub: alirezadir/machine-learning-interviews — ml-system-design.md](https://github.com/alirezadir/machine-learning-interviews/blob/main/src/MLSD/ml-system-design.md)
- [SecondTalent: 8 AI/ML System Design Questions (2026)](https://www.secondtalent.com/interview-guide/ai-ml-engineers-system-design/)
- [IGotAnOffer: ML System Design Interview Guide](https://igotanoffer.com/en/advice/machine-learning-system-design-interview)
- [System Design Handbook: Best ML System Design Resources (2026)](https://www.systemdesignhandbook.com/blog/best-resources-for-ml-system-design-interview/)
- [Index.dev: 50 Principal AI/ML Architect Questions](https://www.index.dev/interview-questions/principal-ai-ml-architect)
