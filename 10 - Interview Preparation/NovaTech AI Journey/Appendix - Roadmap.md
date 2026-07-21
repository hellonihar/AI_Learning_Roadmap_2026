# Appendix: Consolidated Roadmap

## The NovaTech AI Architecture (Maturity Level 3)

```
─────────────────────────────────────────────────────
GOVERNANCE & COMPLIANCE
Model Cards | Bias Audits | Red-Teaming | EU AI Act
─────────────────────────────────────────────────────
ORCHESTRATION LAYER
Intent Routing | Load Balancing | Cost Tracking
─────────────────────────────────────────────────────
SERVING LAYER
vLLM (LLMs) | Triton (non-LLM) | SageMaker/Vertex
─────────────────────────────────────┬───────────────
SHARED INFRASTRUCTURE                   │
├─ Feature Store (Feast + Redis)        │
├─ Model Registry (MLflow)              │ MONITORING
├─ Vector DB (Qdrant)                   │ Prometheus
├─ GPU Cluster (K8s + KubeFlow)         │ Evidently
├─ CI/CD Pipeline (Argo + Great Exp)    │ Arize
└─ Data Lakehouse (Delta Lake on S3)    │
─────────────────────────────────────┴───────────────
DATA FOUNDATION
Ingestion (Kafka) → Bronze → Silver → Gold (Delta Lake)
─────────────────────────────────────────────────────
```

---

## AI Architect Principles (From the Journey)

| Principle | Description | Source |
|-----------|-------------|--------|
| **Start with infrastructure, not models** | Build the factory before manufacturing | Phase 3 — MLOps Foundation |
| **Fail fast, learn faster** | Failure is data for improvement | Phase 2 — The Deep Dive |
| **Know your data distribution** | Training-serving skew kills models silently | Phase 1 → Phase 2 pivot |
| **Build vs. buy with honesty** | Commodity → buy. Differentiation → build | Chapter 8 — The Consultant |
| **Governance is a shield, not a blocker** | Proactive compliance prevents crises | Chapter 18 — Governance Awakening |
| **Business metrics > model metrics** | Accuracy means nothing if users don't adopt | Chapter 19 — Lessons Learned |
| **Start small, scope tightly, prove value** | One use case that works > 17 that don't | Phase 3 — Choosing the Right Problem |
| **RAG before fine-tuning** | Ground in data before memorizing | Phase 4 — RAG Architecture |
| **Quantize early, quantize often** | 4x cost reduction for <3% quality loss | Chapter 17 — The Cost Wake-Up |
| **Platform before autonomy** | Shared infrastructure prevents sprawl | Chapter 16 — Second System Effect |

---

## Technical Terms Glossary

| Term | Definition | First Appears |
|------|------------|---------------|
| **Training-serving skew** | Mismatch between training data distribution and production data distribution | Ch.2 |
| **Label leakage** | Training data contains information not available at inference time (spurious correlations) | Ch.4 |
| **Covariate shift** | Change in the distribution of input features between training and production | Ch.4 |
| **Concept drift** | Change in the relationship between features and labels over time | Ch.9 |
| **Data lakehouse** | Architecture combining data lake flexibility with warehouse ACID guarantees | Ch.5 |
| **Medallion architecture** | Bronze (raw) → Silver (validated) → Gold (curated) data layers | Ch.5 |
| **Delta Lake** | Open-source storage layer for data lakes with ACID transactions | Ch.5 |
| **Feature store** | Central repository for consistent feature computation across training and inference | Ch.6 |
| **Point-in-time join** | Joining features as they existed at prediction time to prevent data leakage | Ch.6 |
| **RAG** | Retrieval-Augmented Generation — retrieve relevant documents before LLM generation | Ch.11 |
| **Dense retrieval** | Embedding-based semantic search using vector similarity | Ch.11 |
| **Hybrid search** | Combining dense (semantic) and sparse (keyword) retrieval | Ch.12 |
| **Cross-encoder** | Re-ranker that compares query-document pairs directly (slow, accurate) | Ch.12 |
| **HNSW** | Hierarchical Navigable Small World — approximate nearest neighbor search algorithm | Ch.13 |
| **Contextual hallucination** | LLM output grounded in retrieved docs but combining facts incorrectly | Ch.14 |
| **Self-verification** | LLM checks its own output against source documents | Ch.14 |
| **Quantization** | Reducing model precision (FP16 → INT4) for memory/speed gains | Ch.17 |
| **MCP** | Model Context Protocol — standard interface for LLM-tool communication | Ch.20 |
| **Ef core** | A/B testing framework for splitting traffic between model versions | Ch.9 |
| **Model registry** | Versioned storage for model artifacts, metadata, and lineage | Ch.9 |
| **Model cards** | Standardized documentation of model purpose, data, metrics, limitations | Ch.18 |
| **Red-teaming** | Adversarial testing to find failure modes before deployment | Ch.18 |
| **Second system effect** | Teams independently building incompatible versions of the same infrastructure | Ch.16 |
| **Federated learning** | Training across distributed data silos without centralizing data | Ch.20 |

---

## The Three Horizons Visual

```
H1: Foundation (M20-M26)      H2: Intelligence (M26-M32)      H3: Autonomy (M32-M40)
─────────────────────────     ─────────────────────────      ─────────────────────────
Unified monitoring            Multi-agent orchestration     60% auto-resolution
Automated retraining          Real-time personalization     Continuous learning
Model catalog                 Multimodal support            Federated learning
Cost optimization             MCP protocol integration      AI CoE as product
```

---

## Recommended Reading (From the Team's Journey)

- **Designing Machine Learning Systems** (Chip Huyen) — the book Elena gave Raj
- **Building Machine Learning Pipelines** (Hapke & Nelson) — Elena's infrastructure bible
- **Machine Learning Engineering** (Andriy Burkov) — Anika's recommendation
- **The Lean Startup** (Eric Ries) — David's framework for AI product development
- **Atlas of AI** (Kate Crawford) — governance team's reference on AI ethics
- **Scaling Monosemanticity** (Anthropic) — what Raj read in Horizon 2 planning

---

*"AI is not a destination. It's a capability you build, fail at, and rebuild."*
