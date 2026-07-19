# Lesson 06.4.6: Managed AI Platforms

## Learning Objectives
- Compare Azure AI Foundry, AWS Bedrock, GCP Vertex AI, and OpenAI Platform
- Evaluate platforms on model selection, RAG support, guardrails, and compliance
- Choose the right platform based on organizational and technical requirements

---

## 1. Why Use a Managed Platform?

Running LLMs in production requires more than an API key. Managed platforms provide:

- **Model hub:** Access to multiple models (proprietary + open-source) from one API
- **Infrastructure:** Auto-scaling, load balancing, GPU management
- **Guardrails:** Built-in content filtering, PII detection, prompt shielding
- **Monitoring:** Usage tracking, cost dashboards, latency alerts
- **Compliance:** Certifications (SOC 2, HIPAA, ISO 27001) out of the box
- **RAG tooling:** Embedding models, vector databases, retrieval pipelines

The tradeoff is vendor lock-in: the deeper you integrate, the harder it is to switch.

---

## 2. Platform Deep-Dives

### 2.1 Azure AI Foundry (Microsoft)

**Model selection:**
- OpenAI models (GPT-4o, o1, o3, embeddings)
- Meta Llama 3, Mistral, Cohere, Phi-3
- HuggingFace model deployment (managed endpoint)
- Model catalog with 1,600+ models

**Strengths:**
- Deep Azure ecosystem integration (Entra ID, VNet, monitor, and compliance)
- Azure AI Search for RAG (hybrid search, semantic ranking)
- Content Safety: multi-modal (text + image), custom severity thresholds
- Responsible AI dashboard: bias detection, interpretability, error analysis
- Enterprise-grade: HIPAA, SOC 2, FedRAMP ready

**Weaknesses:**
- Complex pricing (multiple services, per-region model availability)
- UI can be overwhelming for small teams

**Best for:** Organizations already on Azure, regulated industries, enterprise RAG workloads.

### 2.2 AWS Bedrock (Amazon)

**Model selection:**
- Anthropic Claude 3/4, Meta Llama 3, Cohere, Mistral, Amazon Titan
- Stable Diffusion (via Bedrock)
- No OpenAI models directly (access via SageMaker or API proxy)

**Strengths:**
- Tight AWS integration (IAM, KMS, CloudWatch, VPC)
- Knowledge Bases for RAG: fully managed vector store (Aurora/Pinecone integration)
- Guardrails for Bedrock: denied topics, content filters, PII redaction
- Batch inference for cost savings on large workloads
- Provisioned throughput for predictable performance

**Weaknesses:**
- Smaller model selection (no GPT-4o)
- Guardrails features are newer, less mature than Azure
- RAG knowledge base setup requires multiple AWS services

**Best for:** Organizations already on AWS, applications needing Claude, cost-sensitive batch workloads.

### 2.3 GCP Vertex AI (Google)

**Model selection:**
- Gemini 1.5/2.0 Pro & Flash, Gemini 2.0 Flash
- Anthropic Claude, Meta Llama 3
- Open-source models via Model Garden
- Imagen (image generation)

**Strengths:**
- Gemini Pro: 1M+ token context window (best-in-class for long documents)
- Vertex AI Search: managed RAG with enterprise search
- AutoSxS (side-by-side model evaluation)
- Agent Builder: visual agent construction
- Integration with BigQuery for analytics

**Weaknesses:**
- Smaller ecosystem than AWS/Azure for enterprise compliance
- Gemini quality sometimes lags behind GPT-4o for complex reasoning

**Best for:** Organizations on Google Cloud, long-context document analysis, agent-building workflows.

### 2.4 OpenAI Platform

**Model selection:**
- GPT-4o, GPT-4o-mini, o1, o3, o4-mini
- DALL-E 3 (image generation)
- Whisper (speech-to-text), TTS (text-to-speech)

**Strengths:**
- Best-in-class quality on most benchmarks
- Simplest developer experience (one API key, one endpoint)
- Assistants API: built-in retrieval, code interpreter, function calling
- Rate limits are predictable and generous
- Real-time API (WebRTC) for voice applications

**Weaknesses:**
- No managed RAG storage (bring your own vector DB)
- No built-in guardrails API (use OpenAI Moderation as a separate call)
- No enterprise compliance certifications on standard API (Azure OpenAI for HIPAA etc.)
- Single-provider dependency

**Best for:** Startups, rapid prototyping, applications needing the absolute best model quality, voice apps.

---

## 3. Feature Comparison Matrix

| Feature | Azure AI Foundry | AWS Bedrock | GCP Vertex AI | OpenAI Platform |
|---------|-----------------|-------------|---------------|-----------------|
| GPT-4o access | ✅ Yes | ❌ No (proxy needed) | ❌ No | ✅ Yes |
| Claude access | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |
| Managed RAG | ✅ Azure AI Search | ✅ Knowledge Bases | ✅ Vertex AI Search | ❌ BYO DB |
| Built-in guardrails | ✅ Content Safety | ✅ Guardrails | ✅ (limited) | ⚠️ Moderation API |
| PII detection | ✅ | ✅ | ✅ | ❌ |
| Streaming support | ✅ | ✅ | ✅ | ✅ |
| Batch inference | ⚠️ (Azure Batch) | ✅ Yes | ✅ Yes | ⚠️ (Batch API beta) |
| HIPAA eligible | ✅ Yes | ✅ Yes | ✅ Yes | ❌ (use Azure) |
| Provisioned throughput | ✅ (PTUs) | ✅ Yes | ✅ Yes | ✅ (Provisioned) |
| Multi-region redundancy | ✅ Yes | ✅ Yes | ✅ Yes | ⚠️ (limited) |

---

## 4. Choosing the Right Platform

### Decision Matrix

| If your priority is… | Choose… |
|----------------------|---------|
| Best model quality | OpenAI Platform (GPT-4o + o1) |
| Enterprise compliance & security | Azure AI Foundry |
| Deep AWS integration | AWS Bedrock |
| Long context windows | GCP Vertex AI (Gemini) |
| Fast prototyping | OpenAI Platform |
| Running open-source models | Azure AI Foundry or GCP Vertex AI |
| Multi-model fallback routing | Use a router like LiteLLM across all platforms |

### Multi-Platform Strategy

Many organizations use **more than one platform**:

- **Primary:** Azure AI Foundry for production (compliance, security, RAG)
- **Secondary:** OpenAI Platform for the latest models (o1 for complex reasoning)
- **Fallback:** AWS Bedrock for cost-sensitive workloads (batch Claude)

Unify access with LiteLLM or a custom routing layer:

```python
import litellm

response = litellm.completion(
    model="gpt-4o",  # Or "claude-3-haiku", "vertex_ai/gemini-2.0-flash"
    messages=[{"role": "user", "content": "Hello"}]
)
```

LiteLLM normalizes the API across providers, handles retries, and tracks costs.

---

## 5. Migration Considerations

If you start on one platform and want to switch:

- **OpenAI → Azure:** Nearly identical API. Swap the endpoint and key.
- **OpenAI → Bedrock/Vertex:** Requires rewriting the client library and prompt formatting (Anthropic uses different message formats than OpenAI).
- **RAG portability:** Document indexing pipelines are platform-specific. Use a portable vector DB (Pinecone, Weaviate) to keep retrieval logic neutral.
- **Guardrails:** Platform guardrails are proprietary. Use an open-source layer (NeMo Guardrails) on top to stay portable.

---

## Summary

Each platform makes different tradeoffs. Azure AI Foundry wins on compliance and breadth; OpenAI wins on model quality and simplicity; Bedrock wins for AWS-native teams; Vertex wins for GCP-native teams and long-context tasks. The best strategy for production is often multi-platform: use the right model from the right provider for each task, unified behind a routing layer.

---

## Key Terms
- **Model-as-a-Service (MaaS):** Consuming models via API rather than self-hosting
- **Provisioned throughput:** Reserved model capacity for predictable performance and cost
- **Knowledge Base:** Managed RAG pipeline (ingest → embed → index → retrieve)
- **LiteLLM:** Open-source library that provides a unified API for 100+ LLM providers
- **AutoSxS:** Vertex AI's automated side-by-side model evaluation tool
