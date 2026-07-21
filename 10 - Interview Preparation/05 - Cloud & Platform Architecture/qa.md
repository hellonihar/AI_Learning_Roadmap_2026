# Cloud & Platform Architecture — Q&A

## Q1: Compare AWS Bedrock, Azure AI Foundry, and GCP Vertex AI.

| Aspect | AWS Bedrock | Azure AI Foundry | GCP Vertex AI |
|--------|------------|-------------------|---------------|
| **Base models** | Claude, Llama, Mistral, Titan, Jurassic | OpenAI (GPT-4o, o3), Llama, Mistral, Cohere | Gemini, Claude, Llama, Gemma |
| **Customization** | Fine-tuning, RAG (Knowledge Bases) | Fine-tuning, RAG (AI Search), DPO | Fine-tuning, RAG (Vector Search), RLHF |
| **Serverless** | Yes (pay per token) | Yes (MaaS — pay per token) | Yes (Model-as-a-Service) |
| **Self-hosted** | SageMaker for custom deployment | Managed Compute + K8s | GKE + GPU for custom deployment |
| **Agent framework** | Bedrock Agents | AI Agent Service | Vertex AI Agent Builder |
| **Evaluation** | Bedrock Evaluations | AI Foundry Evaluation | Vertex AI Evaluation |
| **Guardrails** | Bedrock Guardrails | Content Safety + Filter | Vertex AI Safety |
| **Security** | KMS, VPC, PrivateLink | Private Endpoints, RBAC, CMK | VPC-SC, CMEK, IAP |

**Key differentiators:**
- Azure has exclusive OpenAI access (GPT-4o, o3) — best for Microsoft-native shops
- Bedrock has broadest model selection with consistent API
- Vertex AI has strongest data platform integration (BigQuery, Dataflow)

---

## Q2: Design a multi-cloud disaster recovery strategy for AI workloads.

**Architecture:**

```
Primary Cloud (AWS)              DR Cloud (Azure)
┌──────────────────────┐        ┌──────────────────────┐
│  Active Inference     │        │  Warm Standby        │
│  (100% traffic)       │───────►│  (0% traffic, ready)  │
│  ┌───────────────┐   │        │  ┌───────────────┐   │
│  │ Model Replica  │   │        │  │ Model Replica  │   │
│  │ (latest)       │   │        │  │ (latest)       │   │
│  └───────────────┘   │        │  └───────────────┘   │
│  ┌───────────────┐   │        │  ┌───────────────┐   │
│  │ Feature Store  │   │<──────►│  │ Feature Store  │   │
│  │ (primary)      │   │  sync │  │ (replica)      │   │
│  └───────────────┘   │        │  └───────────────┘   │
└──────────────────────┘        └──────────────────────┘
```

**Key decisions:**
- **Active-Active** (both regions serving traffic): Highest cost, lowest RTO (seconds)
- **Active-Passive** (DR region cold/warm): Lower cost, RTO 5-30 minutes
- **Important:** DR should be for the full stack — inference + feature store + model registry + monitoring

**Data replication:**
- Async replication for feature store
- Global model registry (S3 cross-region replication or Artifact Registry multi-region)
- DNS failover (Route53 → Traffic Manager) with health checks

**Fallback behavior:**
- If primary region fails, route traffic to DR
- DR model may use slightly stale features (RPO depends on replication lag)
- Monitor replication lag; trigger alert if exceeds threshold

---

## Q3: What GPU SKU would you choose for fine-tuning a 70B model vs. serving it?

| Task | GPU | Why |
|------|-----|-----|
| **Fine-tune 70B (LoRA)** | 1-2x H100 (80GB) | LoRA + QLoRA fits 70B in 1x H100 with 4-bit quantization. 2x for larger batch size |
| **Fine-tune 70B (Full)** | 8x A100-80GB or 4x H100-80GB | FSDP with ZeRO-3; full params + optimizer states + gradients |
| **Serve 70B (low latency)** | 4x H100-80GB (TP=4) | Tensor parallelism, continuous batching, <100ms per token |
| **Serve 70B (high throughput)** | 8x A100-80GB (TP=8) or 4x H100-80GB | Batch size for throughput; A100 cheaper for inference |
| **Fine-tune 8B (LoRA)** | 1x A10G/L4 (24GB) | Consumer-grade GPU sufficient |
| **Serve 8B** | 1x A10G/L4 or 1x T4 | Quantized (INT4) 8B fits easily |
| **Fine-tune 405B** | 64-128x H100 | Distributed across nodes with FSDP; needs cluster |

**Cost considerations:** H100 is ~3-4x more expensive than A100 but 2-3x faster for training (FP8 support, more TFLOPS). For inference, INT4 quantized models reduce GPU requirements by ~4x.

---

## Q4: How do you secure an AI pipeline end-to-end?

**Data layer:**
- Encryption at rest (AES-256 with CMK) and in transit (TLS 1.3)
- Data masking/PII redaction before training
- Access control with IAM roles (principle of least privilege)
- Audit logging for all data access

**Training layer:**
- Private VPC for training clusters (no public IPs)
- Container image scanning (vulnerability detection)
- Secrets management (Vault/AWS Secrets Manager) — no hardcoded keys
- Training data isolation between clients/tenants

**Model layer:**
- Model artifact signing (prevent tampering)
- Access control on model registry (read-only for inference, write for CI/CD)
- Prompt injection detection and input sanitization
- Output guardrails (toxicity, PII, confidential data)

**Inference layer:**
- Private endpoints (not exposed to internet)
- Rate limiting and authentication per API key
- Request logging without storing PII
- DDoS protection (AWS Shield/CloudFront/Azure WAF)

**Compliance:**
- Audit trail for all model versions and deployments
- Data residency controls (keep EU data in EU)
- Regular penetration testing and red-teaming

---

## Q5: Walk through a cost-optimized architecture for a high-volume inference service.

**Scenario:** 10M requests/day, 70B model, <500ms latency p95.

**Cost breakdown (monthly, estimated):**
| Component | Non-optimized | Optimized | Savings |
|-----------|-------------|-----------|---------|
| GPU compute | $80K (8x A100 on-demand) | $35K (4x H100, spot + reserved) | 56% |
| Networking | $5K | $3K (same-AZ placement) | 40% |
| Storage | $2K (SSD) | $500 (S3 + local cache) | 75% |
| API gateway/LB | $3K | $1K (internal LB) | 66% |

**Optimizations applied:**
1. **Reserved + spot mix:** 40% reserved (baseline), 60% spot (burst). Use spot interruption handling with queue draining.
2. **Model quantization:** INT4 quantized (SqueezeLLM/AWQ) — ~4x memory reduction, minimal quality loss
3. **Prompt caching:** Cache frequent prefixes (system prompts, few-shot). 40% cache hit rate.
4. **KV cache optimization:** GQA + KV cache quantization (FP8). Reduces per-request memory by 50%.
5. **Continuous batching:** vLLM or TensorRT-LLM for max GPU utilization.
6. **Autoscaling:** Scale to zero during low traffic. Hibernation mode — keep model in GPU memory but batch infrequently.
7. **Load shedding:** Queue for overflow, reject with 429 if queue > threshold.

**Net result:** ~60-70% cost reduction while meeting latency SLA.

---

## Q6: Compare spot vs. on-demand vs. reserved GPU instances. When to use each?

| Type | Cost Savings | Stability | Use Case |
|------|------------|-----------|----------|
| On-demand | 0% (baseline) | 100% (always available) | Production inference, critical training |
| Reserved (1yr) | ~30% | 100% | Stable baseline workloads |
| Reserved (3yr) | ~60% | 100% | Long-running training clusters |
| Spot | ~60-90% | Variable (can be reclaimed) | Training (with checkpointing), batch inference |
| Savings Plan | ~20-40% | 100% | Flexible commitment across instance types |

**Best practice:** Use reserved/savings plan for baseline (your minimum 24/7 usage). Use spot for elastic training workloads. Implement checkpoint-restart for spot training — if instance is reclaimed, resume from the latest checkpoint on a new spot instance.

---

## Q7: Design a data lakehouse architecture for ML workloads.

```
┌──────────────────────────────────────────────────────┐
│              ML Applications                           │
│  (Training, Notebooks, Feature Engineering, Serving)  │
├──────────────────────────────────────────────────────┤
│        Data Catalog (Unity/Data Catalog)              │
├──────────────────┬───────────────────────────────────┤
│  Structured      │  Unstructured / Semi-structured    │
│  ┌──────────────┐│  ┌──────────────────────────────┐ │
│  │ Delta Lake   ││  │ Object Store (S3/ADLS/GCS)   │ │
│  │ (Parquet)    ││  │ + Partitioned by date/source  │ │
│  │ ACID, Z-order││  │ + JSON / Images / Video      │ │
│  └──────────────┘│  └──────────────────────────────┘ │
├──────────────────┴───────────────────────────────────┤
│              Ingestion Layer                           │
│  (Kafka for streaming, Airflow/Spark for batch)       │
├──────────────────────────────────────────────────────┤
│              Raw Data Sources                          │
│  (DBs, APIs, Logs, S3 buckets, 3rd party)             │
└──────────────────────────────────────────────────────┘
```

**Key features:**
- **Single copy of data** — both structured and unstructured
- **Schema-on-read** — flexibility for ML experimentation
- **ACID transactions** on structured data (Delta Lake/Iceberg)
- **Time travel** — query data as of any point in time
- **Data quality** — built-in validation (Great Expectations) as part of ingestion

**ML-specific considerations:**
- Store feature engineering code alongside data definitions
- Track data lineage for reproducibility
- Partition training data to avoid full scans
- Use columnar formats (Parquet) for efficient ML reading

---

## Q8: How do you set up RBAC for a multi-team AI platform?

**Role hierarchy:**
| Role | Model Registry | Training | Inference | Data | Approvals |
|------|---------------|----------|-----------|------|-----------|
| Admin | Read/Write/Delete | Full | Full | Full | Yes |
| ML Engineer | Read/Write | Create/Submit | Read | Read training | No (self-serve) |
| MLE/SRE | Read | Read | Deploy/Scale | Read | Yes (deploy) |
| Data Scientist | Read/Write | Create/Submit | Read | Read | No |
| Data Engineer | Read features | No | No | Write | No |
| Auditor | Read only | Read logs | Read logs | Read | No |
| Viewer | Read | No | No | Read | No |

**Implementation:**
- Use cloud IAM (AWS IAM / Azure RBAC) for infrastructure
- Use K8s RBAC for pod-level access
- Use model registry ACLs for model-level permissions
- Tag all resources with team, environment, cost-center for cost allocation

---

## Q9: What networking considerations are critical for multi-node GPU training?

**Bandwidth hierarchy:**
- **Within GPU (HBM):** 2-3 TB/s
- **Within node (NVLink):** 600-900 GB/s (all-to-all GPU communication)
- **Between nodes (InfiniBand):** 200-400 Gbps per link (NVIDIA uses 4x 400G IB per node)
- **Between regions:** Limited (50-100 Gbps) — avoid cross-region training

**Key requirements:**
- All GPU nodes in the same rack or adjacent racks (minimize hop distance)
- InfiniBand or at least EFA (Elastic Fabric Adapter) for gradient sync
- NCCL tuning: NCCL_ALGO=Ring, NCCL_PROTO=Simple, NCCL_MIN_NCHANNELS
- Network topology-aware scheduling (K8s topology manager, NUMA alignment)
- Avoid oversubscription: each node should have dedicated IB bandwidth
- TCP/IP for model artifact loading (not for gradient sync)

**Anti-patterns:**
- Training across data centers (latency too high for gradient sync)
- Mixing GPU generations in same training job (H100 + A100 → straggler effect)
- Using Ethernet for gradient sync (200x slower than IB)

---

## Q10: Compare managed ML platforms vs. self-managed Kubernetes for AI.

| Aspect | Managed (SageMaker/Azure ML/Vertex AI) | Self-Managed (EKS/AKS/GKE + KubeFlow) |
|--------|---------------------------------------|---------------------------------------|
| Setup time | Minutes | Weeks |
| Ops overhead | Low (managed infra) | High (cluster management) |
| Flexibility | Limited by platform APIs | Full control |
| Cost (dev) | Lower (no infra team) | Higher (need dedicated ops) |
| Cost (prod) | Higher (platform markup) | Lower (raw compute pricing) |
| GPU access | Good (managed pools) | Variable (depends on provider) |
| Custom networking | Limited | Full control (VPC, IB) |
| Best for | Teams without infra expertise | Teams with strong infra, large scale |

**Hybrid approach:** Use managed platform for development/experimentation, self-managed K8s for production training at scale.

---

## Q11: How would you migrate ML workloads from one cloud to another?

**Phased approach:**

1. **Audit** — inventory all ML workloads, dependencies, data stores
2. **Provider-agnostic layer** — abstract cloud APIs behind a compatibility layer (Kubernetes, MLflow, S3-compatible storage)
3. **Data migration** — sync data to new cloud (cross-region replication, then cut over)
4. **Model migration** — retrain or transfer model artifacts (ONNX or SageMaker NEFF → TensorRT)
5. **Inference migration** — deploy to new cloud in shadow mode, compare results, cut over
6. **Decommission** — keep old cloud as DR for N months, then fully decommission

**Key challenges:**
- GPU availability varies by cloud (H100 in Azure vs. GCP)
- Cloud-specific ML services (Bedrock → Vertex AI migration requires code changes)
- Data egress costs (can be $5-10K+ per TB)
- Networking latency between old and new cloud during migration

---

## Q12: Design a training infrastructure on Azure with AI Foundry.

**Architecture:**
```
Azure AI Foundry
      │
      ├── AI Studio (experimentation, notebooks)
      ├── Prompt Flow (RAG, evaluation)
      ├── Model Catalog (curated + custom models)
      │
      ├── Training: Azure ML + ND-series VMs (H100/NVLink)
      │     ├── Distributed training with FSDP
      │     └── Azure ML Pipelines (orchestration)
      │
      ├── Serving: Azure ML Managed Endpoint (MaaS)
      │     └── or AKS with GPU node pool (self-managed)
      │
      ├── Data: Azure Data Lake Storage Gen2 + AI Search (RAG)
      │
      └── Governance: Azure Policy + Purview + Defender
```

**Key Azure-specific considerations:**
- Use Azure Managed Lustre for high-throughput checkpoint storage
- Azure AI Search for RAG with hybrid (vector + keyword) retrieval
- OpenAI models available exclusively via Azure (global standard + PTU)
- RBAC via Azure AD with Managed Identity for service-to-service auth
- Cost tracking via Azure Cost Management + tags per project

---

## External Resources

- [Index.dev: 50 Principal AI/ML Architect Questions](https://www.index.dev/interview-questions/principal-ai-ml-architect)
- [Interview Guys: 10 AI Solutions Architect Questions (2026)](https://blog.theinterviewguys.com/ai-solutions-architect-interview-questions/)
- [Future Skills Academy: Top AI Architect Questions (2026)](https://futureskillsacademy.com/blog/top-ai-architect-interview-questions-and-answers/)
- [GitHub: aakriti1318/interview_questions/AI_Architect](https://github.com/aakriti1318/interview_questions/tree/main/AI_Architect)
- [CVOwl: 20 AI Systems Architect Questions](https://www.cvowl.com/blog/ai-systems-architect-interview-questions-answers)
