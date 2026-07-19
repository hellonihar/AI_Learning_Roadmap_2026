# Lesson 06.4.3: Observability for LLM Applications

## Learning Objectives
- Define and measure the key metrics for LLM applications
- Implement quality monitoring, hallucination detection, and user feedback tracking
- Use observability tools (LangSmith, W&B Prompts, Helicone, MLflow)
- Add distributed tracing for RAG, tool calls, and multi-step chains

---

## 1. Why Observability Matters for LLMs

Traditional application monitoring (CPU, memory, error rate) is necessary but insufficient for LLM apps. You must also monitor:

- **Semantic quality:** Is the response good?
- **Cost:** Each inference costs real money.
- **Safety:** Is the model generating harmful or leaked content?
- **Latency breakdown:** Where is time spent — embedding, retrieval, generation, tool calls?

A monitoring gap of even minutes can cost thousands of dollars or erode user trust through bad responses.

---

## 2. Core Metrics

### Operational Metrics

| Metric | What It Measures | Alert Threshold |
|--------|-----------------|-----------------|
| Time to First Token (TTFT) | Latency before first token appears | > 2s |
| Tokens per Second (TPS) | Generation speed | < 20 t/s |
| End-to-end latency | Total request time | > 10s |
| Throughput | Requests per minute | Depends on SLA |
| Error rate | 4xx/5xx, timeouts | > 1% |
| GPU utilization | % GPU used | < 30% (under-utilized) or > 95% (bottleneck) |

### Cost Metrics

- **Cost per request** = (input_tokens × input_price) + (output_tokens × output_price)
- **Daily/weekly burn rate** — project monthly costs
- **Cost by model** — track which models are driving spend
- **Cost by user/feature** — attribute costs to teams or features

### Quality Metrics

- **Response quality score:** Use an LLM-as-a-judge (GPT-4o, Claude) to rate responses on relevance, helpfulness, and safety. Score 1–5.
- **User feedback score:** Aggregate explicit (thumbs up/down) and implicit signals.
- **Hallucination rate:** % of responses containing factual errors (detected by fact-checking against retrieved documents or knowledge base).
- **Grounding score:** For RAG apps, does the response cite the retrieved context? Measure citation precision and recall.

---

## 3. Hallucination Detection

Techniques to detect when the model is making things up:

| Method | How It Works | Latency | Accuracy |
|--------|-------------|---------|----------|
| Self-consistency | Generate N responses, check agreement | High (N× cost) | High |
| Logit analysis | Low confidence in output tokens = hallucination risk | Low | Medium |
| External fact-check | Compare claims against known database | Medium | High |
| LLM-as-judge | Ask another model to check for hallucination | Medium | High |

**Recommendation:** Start with logit-based detection (zero added latency). For high-stakes apps, add LLM-as-judge with a lightweight model (GPT-4o-mini).

---

## 4. Observability Tools

### LangSmith (LangChain)
- **Tracing:** Automatic spans for every LLM call, tool call, retriever, chain step.
- **Evaluation:** Run datasets against models, compare runs.
- **Monitoring:** Latency, cost, feedback dashboards.
- **Hub:** Share and version prompts.

### Weights & Biases Prompts
- **Prompt tracing:** Track prompt versions, model responses, and metadata.
- **Experimentation:** A/B test prompts, compare outputs.
- **Collaboration:** Share traces with team members.

### Helicone
- **Proxy-based:** Drop-in replacement for OpenAI/Azure endpoint.
- **Metrics:** Cost, latency, token usage per user/session.
- **Caching:** Built-in exact and semantic caching.
- **Alerting:** Slack/webhook alerts on cost spikes or error rate changes.

### MLflow (v2.15+)
- **Tracing:** Native LLM tracing via MLflow Tracing.
- **Model registry:** Track which model version is deployed.
- **Evaluation:** Built-in LLM eval metrics.

---

## 5. Distributed Tracing for LLM Apps

Traditional distributed tracing (OpenTelemetry) extends naturally to LLM workloads.

### Span Types for RAG

```
User Query
  ├─ [SPAN] Embedding: query → vector
  ├─ [SPAN] Vector Search: vector → chunks
  ├─ [SPAN] LLM Call: prompt + chunks → response
  └─ [SPAN] Post-processing: response → final output
```

### Span Types for Agent / Tool Use

```
User Query
  ├─ [SPAN] LLM: route query → tool decision
  ├─ [SPAN] Tool: search_web(query) → results
  ├─ [SPAN] Tool: calculator(expression) → result
  ├─ [SPAN] LLM: synthesize results → response
  └─ [SPAN] Guard: check response for PII
```

### What to capture in each span:
- `input` / `output` (with PII redaction)
- `token_count` (prompt + completion)
- `model_name` and `provider`
- `latency_ms`
- `error` (if any)
- `tags` (user_id, session_id, feature_name)

### Implementation

```python
# LangSmith auto-traces LangChain:
from langsmith import Client
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "..."

# For non-LangChain code, use @traceable decorator:
from langsmith import traceable

@traceable
def my_step(input_data):
    ...
    return output
```

For custom implementations, emit OpenTelemetry spans and ingest via Jaeger, Grafana Tempo, or Datadog.

---

## 6. Building an Observability Stack

```
Application → OpenTelemetry Collector → [Metrics: Prometheus]
                                         [Traces: Jaeger/Tempo]
                                         [Logs: Loki/Elasticsearch]
                                              ↓
                                       Grafana Dashboard
```

Or use an all-in-one LLM observability platform (LangSmith, Helicone) for faster setup.

**Minimum viable observability:**
1. Log every LLM request and response to a database.
2. Track token usage and cost per request.
3. Collect user feedback (thumbs up/down).
4. Set alerts on error rate and cost spikes.

---

## Summary

Observability for LLMs combines traditional metrics (latency, error rate) with LLM-specific ones (hallucination rate, cost per token, grounding score). Tools like LangSmith and Helicone provide turnkey tracing. Start with the minimum—log all requests, track costs, collect feedback—then layer in semantic monitoring as your app matures.

---

## Key Terms
- **TTFT (Time to First Token):** Latency from request submission to the first output token
- **Hallucination rate:** Percentage of responses that contain unsupported or false claims
- **LLM-as-judge:** Using a language model to evaluate the quality of another model's response
- **Span:** A named, timed operation in a trace (e.g., "vector_search", "llm_call")
- **Grounding score:** A measure of how well a response is supported by the retrieved context
