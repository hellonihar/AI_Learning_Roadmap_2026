# MLOps & Production — Q&A

## Q1: How do you detect and handle data drift in production?

**Detection methods:**
- **Statistical tests** — Kolmogorov-Smirnov, Chi-squared, Wasserstein distance between reference and current data distribution
- **Population stability index (PSI)** — measures shift in score distributions
- **Feature-level monitoring** — track mean, std, null ratio, min/max per feature per time window
- **Model-level monitoring** — track prediction distribution shift (even if labels aren't available)

**Handling drift:**
1. **Alert** — detect drift, notify team, log samples
2. **Diagnose** — which features drifted? Is it covariate shift (input dist) or concept shift (P(y|x) changed)?
3. **Respond:**
   - **Minor drift** — continue monitoring, log for retraining dataset
   - **Moderate drift** — trigger automated retraining pipeline with recent data
   - **Severe drift** — rollback to previous known-good model, investigate root cause
4. **Automated retraining:** trigger when drift exceeds threshold for N consecutive windows

**Tools:** WhyLabs, Arize AI, Evidently AI, NannyML, MLflow.

---

## Q2: Design a CI/CD pipeline for an ML system with automated retraining.

```
Git Push → Linting/Tests → Train Trigger → Data Validation → Training → Eval → Promote → Deploy
                                                              ↓
                                                        Model Registry
```

**Stages:**

1. **Code push** — model code, training config, orchestration DAGs
2. **CI (continuous integration):**
   - Unit tests: data processing, feature engineering, inference code
   - Integration tests: end-to-end on a small dataset
   - Model tests: overfitting check on small data, NaN gradient check
3. **CD (continuous delivery):**
   - Training trigger: scheduled (weekly) or event-driven (drift detected, new data available)
   - Data validation: Great Expectations suite (schema, range, null checks)
   - Training: distributed training job on preemptible GPUs
   - Evaluation: compare against current production model on holdout set
   - **Promotion gate:** must beat current model on key metrics (not just accuracy — also latency, memory)
4. **Deployment:**
   - Canary — 5% → 25% → 100% with automated rollback
   - Shadow mode — run alongside current model, compare offline
   - A/B test — if statistical significance achieved, full rollout

**Key principle:** Every GATE must be automated and auditable. No manual approvals for production promotion.

---

## Q3: What metrics would you monitor for a production LLM service?

**Infrastructure:**
- **Latency:** TTFT (time to first token), TPOT (time per output token), end-to-end latency (p50, p95, p99)
- **Throughput:** requests per second, tokens per second
- **GPU utilization:** compute utilization, memory utilization, memory bandwidth
- **Error rate:** 4xx, 5xx, timeout rate
- **Queue depth:** number of requests waiting for inference

**Model quality (offline):**
- Perplexity, accuracy on held-out eval sets
- Toxicity/content safety scores
- Bias evaluation (stereotype benchmark)

**Model quality (online):**
- **User feedback:** thumbs up/down, ratings, explicit feedback
- **Retry rate:** users rephrasing questions (signal of poor responses)
- **Escalation rate:** percentage of conversations requiring human handoff
- **Hallucination rate:** sampled human evaluation (with automated proxy metrics)

**Cost:**
- Cost per request, cost per token, GPU hours per model
- Cache hit rate (prompt caching, KV cache)
- Input/output token ratio

---

## Q4: How would you roll back a bad model deployment with zero downtime?

**Strategy: Blue-Green Deployment**

1. **Maintain two environments:** Blue (current production) and Green (new version)
2. **Deploy new model to Green** — warm it up with real traffic (shadow mode)
3. **Health checks** — automated: latency, error rate, prediction distribution shift
4. **Traffic switch** — gradually shift % of traffic to Green
5. **Rollback mechanism:**
   - If error rate increases by >10%, automatically revert to Blue (100%)
   - If prediction distribution shows unexpected shift, trigger human review
   - Rollback by switching DNS/load balancer back to Blue (instant, no redeploy needed)

**Key components:**
- Feature flags (LaunchDarkly or similar) for instant traffic control
- Model registry keeps all versions with metadata (deployment date, performance metrics)
- Automated canary analysis: if confidence interval for key metric doesn't show improvement, abort

---

## Q5: Compare serverless vs. dedicated GPU serving for cost and latency.

| Aspect | Serverless (e.g., AWS SageMaker Serverless, Beam) | Dedicated GPU (vLLM, TGI on K8s) |
|--------|---------------------------------------------------|-----------------------------------|
| Cold start | 5-30 seconds (model load from disk) | Warm (pre-loaded) |
| Latency (hot) | Higher (warm-up variability) | Lower and consistent |
| Cost at low volume | Very cheap (pay per inference) | Expensive (pay for idle GPU) |
| Cost at high volume | Expensive (no volume discounts) | Cheap (amortized over requests) |
| GPU choice | Limited (usually A10G, L4) | Full range (H100, A100, B200) |
| Auto-scaling | Instant | Needs HPA, node pool scaling (5-10 min) |
| Best for | Prototyping, low volume, variable traffic | Production, high volume, consistent traffic |

**Hybrid:** Use serverless for burst traffic and dedicated for baseline. Route excess traffic to serverless during spikes.

---

## Q6: Design a canary deployment strategy for a new model version.

1. **Shadow mode** (pre-canary): New model runs alongside current model but doesn't serve users. Compare predictions offline. Duration: 2-7 days.

2. **Canary stages:**
   - Stage 1: 5% of traffic, 24h monitoring
   - Stage 2: 25% of traffic, 48h monitoring
   - Stage 3: 75% of traffic, 24h monitoring
   - Stage 4: 100% rollout

3. **Monitoring gates at each stage:**
   - **Technical:** latency delta < 10%, error rate < current, memory stable
   - **Business:** no significant drop in user engagement metrics
   - **Quality:** prediction distribution within expected range (drift test)

4. **Automated decisions:**
   - Pass → proceed to next stage
   - Fail → rollback to 0%, log data for root cause analysis
   - Inconclusive → extend observation period

5. **User segmentation:** Route based on user_id hash. Users stay in same model for consistent experience during the experiment.

---

## Q7: How do you set up alerting for model degradation?

**Alert hierarchy:**

1. **Technical alerts (immediate, P0/P1):**
   - Service down / high error rate > 5%
   - Latency p99 > SLA threshold
   - GPU memory OOM
   - These trigger immediate on-call via PagerDuty/OpsGenie

2. **Quality alerts (moderate, P2):**
   - Prediction distribution shift (PSI > 0.2)
   - Feature drift detected (KS test p-value < 0.01 for N windows)
   - Model confidence decreasing over time
   - Sent to daily digest, automated retraining trigger

3. **Business alerts (strategic, P3):**
   - User satisfaction metric dropped by >5%
   - Escalation rate increased by >10%
   - A/B experiment shows regression
   - Sent to weekly review, product team notified

**Config tips:**
- Don't alert on single data points — use rolling windows
- Alert on rate of change, not absolute values (seasonality handling)
- Establish baseline during model validation, alert on deviation

---

## Q8: Explain a complete MLOps platform architecture supporting 50+ teams.

**Platform layers:**

```
┌──────────────────────────────────────────────────────┐
│                  User Interface                        │
│  (MLflow UI, Jupyter, Kubeflow Dashboard, W&B)       │
├──────────────────────────────────────────────────────┤
│           Workflow Orchestration                       │
│  (Airflow, Kubeflow Pipelines, Prefect, Dagster)      │
├─────────────────────┬────────────────────────────────┤
│   Training Platform  │     Serving Platform            │
│  (K8s + GPU, Ray,   │  (vLLM, TGI, Triton, K8s,      │
│   FSDP, DeepSpeed)   │   Istio, Knative)               │
├──────────┬──────────┴──────────┬───────────────────────┤
│ Feature  │ Model Registry      │ Monitoring             │
│ Store    │ (MLflow, S3,       │ (Prometheus/Grafana,   │
│ (Feast,  │  versioning,        │  Evidently, WhyLabs,   │
│  Tecton) │  lineage)           │  Arize)                │
├──────────┴─────────────────────┴───────────────────────┤
│                  Infrastructure                          │
│  (K8s, GPU Nodes, Istio, S3/ADLS, VPC, IAM)            │
└────────────────────────────────────────────────────────┘
```

**Multi-tenancy:**
- Each team gets a namespace with resource quotas
- Shared model registry (read all, write own)
- Self-service model deployment via PRs (GitOps)

**Key principle:** Platform teams build the foundation, ML teams retain autonomy over their models and pipelines.

---

## Q9: How do you manage model versioning and lineage across environments?

**Model Registry (MLflow/W&B/Neptune):**
- Every training run logs: params, metrics, artifacts, environment (conda/pip freeze), dataset hash
- Models are versioned with semantic versioning (or hash-based)
- Model card documents: training data, evaluation results, intended use, limitations

**Lineage tracking:**
- Data source (table/object path + timestamp)
- Training code (git commit hash)
- Hyperparameters and config
- Evaluation dataset and metrics
- Deployment target and timestamp
- Monitoring dashboard link

**Cross-environment promotion:**
- Dev → Staging → Prod (each promotion requires passing evaluation gates)
- Each promotion is recorded with timestamp and approver
- Rollback = redeploy previous registry version (immutable version tags)

---

## Q10: Design an automated retraining pipeline with data quality validation gates.

**Pipeline stages:**

1. **Trigger** — scheduled (weekly) or event-driven (drift detected, new labeled data available)
2. **Data extraction** — pull new training data from data warehouse (last N days)
3. **Data validation:**
   - Schema validation (column names, dtypes, null constraints)
   - Range checks (no outlier features outside expected bounds)
   - Label quality (distribution consistent with historical)
   - Data freshness (minimum row count, maximum age)
   - If validation fails → alert, skip retraining, log report
4. **Feature computation** — run feature engineering pipeline, validate feature distributions
5. **Training** — launch distributed training with automatic hyperparameter optimization
6. **Evaluation:**
   - Compare against current production model
   - Metric thresholds: must outperform or be within 2% on key metrics
   - Additional tests: fairness evaluation, slice-based evaluation
7. **Model card update** — regenerate model card with new training data stats
8. **Registry** — if passing, register new version as "staging"
9. **Deployment** — proceed to CI/CD pipeline for staged rollout

---

## Q11: How do you handle feedback loops in production ML systems?

**Feedback loops occur when model predictions affect future data distribution, causing the model to reinforce its own biases.**

**Example:** Fraud detection model blocks users who look fraudulent. Those users never transact, so the model lacks counter-examples, becoming increasingly aggressive.

**Mitigation strategies:**

1. **Exploration traffic** — randomly show alternative decisions to a small % of traffic (e.g., epsilon-greedy)
2. **Holdout data** — maintain a clean, non-model-influenced dataset for evaluation
3. **Counterfactual logging** — log what the model WOULD have predicted even when overridden by business rules
4. **Periodic retraining with fresh data** — include data from after the model's deployment
5. **Separate exploration and exploitation** — dedicated exploration model, trained specifically to gather diverse data
6. **Awareness in metric design** — don't optimize for engagement in recommendation without controlling for diversity

---

## Q12: What infrastructure would you use for distributed training on Kubernetes?

**Architecture:**

```
┌──────────────────────────────────────────┐
│           Kubernetes Cluster              │
│                                           │
│  ┌──────────────┐    ┌──────────────┐    │
│  │  GPU Node     │    │  GPU Node     │    │
│  │  Pool (H100)  │    │  Pool (H100)  │    │
│  ├──────────────┤    ├──────────────┤    │
│  │ 8x H100      │    │ 8x H100      │    │
│  │ NVLink 4.0   │    │ NVLink 4.0   │    │
│  └──────────────┘    └──────────────┘    │
│           │                   │           │
│           └────────┬──────────┘           │
│                    │ InfiniBand           │
│  ┌─────────────────┴─────────────────┐   │
│  │   Storage: NVMe (local) +        │   │
│  │   Parallel FS (Lustre/GPUDirect)  │   │
│  └───────────────────────────────────┘   │
└──────────────────────────────────────────┘
```

**Components:**
- **KubeFlow** for pipeline orchestration or **Ray** for distributed training
- **Volcano/Kueue** for gang scheduling (all GPUs allocated at once)
- **Training operator** — PyTorchJob (KubeFlow) or RayJob
- **Node pools:** spot instances for training, on-demand for critical jobs
- **Storage:** CSI driver for parallel filesystem, local SSD for data cache
- **Monitoring:** Prometheus + DCGM Exporter (NVIDIA GPU metrics) + Grafana

**Key Kubernetes configurations:**
- NUMA-aware scheduling (GPU/CPU affinity)
- HugePages for memory efficiency
- Pod security policies for container permissions
- Network policies for inter-node communication

---

## External Resources

- [SecondTalent: 8 AI/ML System Design Questions (2026)](https://www.secondtalent.com/interview-guide/ai-ml-engineers-system-design/)
- [Index.dev: 50 Principal AI/ML Architect Questions](https://www.index.dev/interview-questions/principal-ai-ml-architect)
- [KORE1: AI/ML Engineer Interview Questions (2026)](https://www.kore1.com/ml-engineer-interview-questions)
- [System Design Handbook: ML System Design Guide (2026)](https://www.systemdesignhandbook.com/guides/ml-system-design/)
- [GitHub: alirezadir/machine-learning-interviews — ml-system-design.md](https://github.com/alirezadir/machine-learning-interviews/blob/main/src/MLSD/ml-system-design.md)
