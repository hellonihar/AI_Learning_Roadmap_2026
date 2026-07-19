# LLMOps Cheatsheet

## 1. UX Patterns

| Pattern | Use Case | Key Design Principle |
|---------|----------|---------------------|
| Chat | Open-ended Q&A, support | Editable history, regenerate, bot indicators |
| Copilot | Writing, coding, design | Invoke on demand, ghost text, user stays driver |
| Autonomous Agent | Multi-step tasks, automation | Transparent reasoning, pause/confirm gates |

**Uncertainty:** Show confidence scores (< threshold → ask user). Cite sources. Let the model say "I don't know."

**Streaming:** Use SSE/WebSocket → render as tokens arrive → stable layout → stop button.

**Feedback:** Thumbs up/down per response + free-text follow-up on downvote. Track copy-paste, regeneration rate as implicit signals.

---

## 2. Deployment Options

| Approach | Tools | When to Use |
|----------|-------|-------------|
| Self-hosted | vLLM, TGI, Ollama, llama.cpp | Data privacy, high volume, customize inference |
| Managed (API) | OpenAI, Anthropic, Bedrock, Azure | Speed to market, spiky traffic, auto-updates |
| Hybrid | Router + mix of self/managed | Cost-quality tradeoff, fallback strategy |

**Scaling:** Horizontal > vertical. Use queues (RabbitMQ, Redis Streams) for back-pressure.
**Rate limiting:** Token bucket, sliding window, concurrent limit. Implement at API gateway.
**Caching:** Exact match for FAQ. Semantic (vector similarity) for chat; can reduce costs 30-60%.
**Load balancing:** Route by complexity, user tier, or model fallback chain.

---

## 3. Observability Metrics

| Category | Key Metrics | Alerts |
|----------|-------------|--------|
| Operations | TTFT, TPS, end-to-end latency, throughput, error rate | TTFT > 2s, error rate > 1% |
| Cost | Cost/request, daily burn, cost by model/user | $ spike > 20% day-over-day |
| Quality | Response quality score (1-5), hallucination rate, grounding score | Quality < 3, hallucination > 5% |
| User feedback | Thumbs up/down ratio, regeneration rate | Downvote rate > 10% |

**Hallucination detection:** Self-consistency (high cost, high accuracy) → Logit analysis (zero latency, medium accuracy) → LLM-as-judge (medium cost, high accuracy).

**Tools:**
- **LangSmith:** Tracing + eval + monitoring (LangChain-native)
- **W&B Prompts:** Prompt versioning + experimentation
- **Helicone:** Proxy-based metrics + caching + alerting
- **MLflow:** Tracking + registry + eval

**Span types:** `llm`, `retriever`, `tool`, `chain`, `embedding`. Capture input, output, token count, latency, model name, error.

---

## 4. Guardrail Types

| Layer | Check | Tools |
|-------|-------|-------|
| Input guard | Prompt injection, jailbreak | NeMo Guardrails, Guardrails AI, regex patterns |
| Content moderation | Hate, violence, self-harm, sexual | OpenAI Moderation API, Azure Content Safety, Llama Guard |
| PII detection | Email, SSN, phone, credit card | Microsoft Presidio, regex, NER models |
| Output guard | Harmful content, PII leakage | Same as input guard + custom LLM-as-judge |
| Hallucination | Factual consistency | Self-consistency, external fact-check, LLM-as-judge |

**Defense in depth:** All four layers (input → moderation → PII → output). Never rely on a single guardrail.

**Testing:** Garak (automated red-teaming), PyRIT (adversarial prompts), Counterfit.

---

## 5. Governance Checklist

- [ ] Model card published
- [ ] System card published
- [ ] Acceptable Use Policy documented
- [ ] Audit trail (immutable logs)
- [ ] Data retention policy
- [ ] Bias evaluation (disaggregated)
- [ ] Transparency disclosure
- [ ] Human review process defined
- [ ] Copyright compliance
- [ ] Regulatory mapping (EU AI Act tier)

**Key regulations:** EU AI Act (tiered, enforcement 2025-2027), US Executive Order on AI (safety testing, NIST standards), China AI Regulation (algorithm registration). Copyright lawsuits pending: NYT v. OpenAI, Getty v. Stability AI.

---

## 6. Platform Comparison

| Feature | Azure AI Foundry | AWS Bedrock | GCP Vertex AI | OpenAI Platform |
|---------|-----------------|-------------|---------------|-----------------|
| GPT-4o | ✅ | ❌ | ❌ | ✅ |
| Claude | ✅ | ✅ | ✅ | ❌ |
| Managed RAG | ✅ Search | ✅ KB | ✅ Search | ❌ BYO |
| Guardrails | ✅ Content Safety | ✅ Guardrails | ⚠️ Limited | ⚠️ Moderation |
| HIPAA | ✅ | ✅ | ✅ | ❌ |
| Best for | Enterprise, regulated | AWS-native, Claude | Long-context, GCP-native | Quality, speed |

Consider multi-platform with LiteLLM as a unified routing layer.

---

## Quick API Reference

```python
# LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"

# @traceable decorator
from langsmith import traceable
@traceable(run_type="llm", tags=["production"])
def my_llm_call(prompt: str) -> str: ...

# OpenAI Moderation
from openai import OpenAI
client = OpenAI()
result = client.moderations.create(input="text")
is_flagged = result.results[0].flagged

# LiteLLM multi-provider
import litellm
resp = litellm.completion(
    model="gpt-4o",  # or "claude-3-haiku", "vertex_ai/gemini-2.0-flash"
    messages=[{"role": "user", "content": "Hi"}]
)

# Semantic cache (pseudocode)
cache = SemanticCache(threshold=0.92, ttl=3600)
if cached := cache.get(query):
    return cached
else:
    response = llm(query)
    cache.set(query, response)
    return response
```
