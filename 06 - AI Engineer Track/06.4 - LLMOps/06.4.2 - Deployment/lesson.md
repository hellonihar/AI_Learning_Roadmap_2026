# Lesson 06.4.2: Deploying LLMs

## Learning Objectives
- Compare self-hosted and managed deployment options
- Implement horizontal scaling, rate limiting, and request queuing
- Use semantic caching to reduce cost and latency
- Load-balance across multiple models for cost-quality tradeoffs

---

## 1. Self-Hosted vs Managed

The first decision: do you run models yourself or call an API?

### Self-Hosted (On Your Infrastructure)

| Tool | Framework | Strengths | Weaknesses |
|------|-----------|-----------|------------|
| **vLLM** | Python / C++ | High throughput via PagedAttention, continuous batching | GPU memory-heavy, requires CUDA |
| **TGI** (Text Generation Inference) | Rust / Python | HuggingFace ecosystem, token streaming, tensor parallelism | Still maturing, fewer model optimizations than vLLM |
| **Ollama** | Go | Local dev, simple API, CPU-friendly with quantization | Not for production scale (single-node) |
| **llama.cpp** | C++ | Runs on CPU/Apple Silicon, quantization (GGUF) | No built-in batching or serving |

**When to self-host:**
- You need full data privacy (no data leaves your network)
- You have predictable, high-volume traffic (savings > infra cost)
- You need to customize the inference stack (quantization, speculative decoding)

### Managed (API-Based)

| Provider | Models | Pricing Model | Key Feature |
|----------|--------|---------------|-------------|
| OpenAI | GPT-4o, o1, o3 | Per-token (input/output) | Highest quality, broadest capabilities |
| Anthropic | Claude 3.5, Claude 4 | Per-token | Long context (200K tokens), safety focus |
| AWS Bedrock | Titan, Llama, Claude | Per-token + provisioned | VPC integration, already on AWS |
| Azure OpenAI | GPT-4o, o1 | Per-token + reserved | Enterprise compliance, Entra ID auth |

**When to go managed:**
- Speed to market matters (no infra setup)
- Traffic is spiky or unpredictable
- You want automatic updates to the latest model versions

---

## 2. Scaling Strategies

### Horizontal Scaling
Scale out, not up. Run multiple replicas behind a load balancer.

```
User → LB → Queue → [Worker 1, Worker 2, …, Worker N] → Model
```

**Auto-scaling triggers:**
- Queue depth (number of pending requests)
- GPU utilization > 80%
- P50/P95 latency exceeding a threshold

### Request Queuing
LLM inference is slow (seconds). Queuing prevents thundering herds and provides back-pressure.

- **Use a message queue:** RabbitMQ, Redis Streams, or SQS.
- **Prioritization:** Interactive requests (chat) get priority over batch (summarization).
- **Timeouts:** Hard timeout per request (e.g., 60s) to prevent stuck workers.
- **Graceful degradation:** If queue depth exceeds N, return a "busy" response or fall back to a smaller model.

### Rate Limiting
Protect downstream APIs (OpenAI) and your own GPU nodes.

| Strategy | How It Works | Use Case |
|----------|-------------|----------|
| Token bucket | Fixed refill rate, burst capacity | Smooth traffic spikes |
| Sliding window | Max requests per sliding time window | Hard API limits |
| Concurrent limit | Max in-flight requests | GPU memory constraints |

Implement at the API gateway (Envoy, Kong) or application middleware.

---

## 3. Caching

LLM inference is expensive. Caching avoids redundant work.

### Exact Match Caching
Store `{prompt → response}` for identical prompts. Works well for:
- System prompts + temperature=0 queries
- FAQ-style questions
- Template-based generation

### Semantic Caching
Cache responses for semantically similar prompts, not just identical ones.

**How it works:**
1. Incoming query → embedding vector (e.g., text-embedding-3-small)
2. Search cache (vector DB) for nearest neighbor above similarity threshold
3. If found → return cached response; if not → call LLM and cache result

**Tools:** Redis with RedisVL, pgvector, Qdrant, Chroma.

**Key parameters:**
- Similarity threshold (typical: 0.85–0.95)
- Cache TTL (varies by use case; shorter for dynamic content)
- Cache invalidation (clear on model update or content change)

Semantic caching can reduce LLM costs by 30–60% for chat applications.

---

## 4. Load Balancing Across Models

Not all requests need GPT-4. Route intelligently:

```
                  ┌─ GPT-4o (high quality, high cost)
User → Router → ──┼─ GPT-4o-mini (medium quality, low cost)
                  └─ Self-hosted Llama-3 (low quality, lowest cost)
```

**Routing logic examples:**
- **Prompt complexity:** Short queries → small model; multi-step reasoning → large model
- **User tier:** Free tier → small model; Pro tier → large model
- **Fallback chain:** Try small model first; if confidence < threshold, escalate to larger model
- **A/B testing:** Gradually shift % traffic to a new model and compare quality scores

**Frameworks:** LiteLLM (unified API for 100+ providers), OpenRouter, Portkey.

---

## 5. Deployment Checklist

- [ ] Canary deployment: roll out new models to 5% of traffic first
- [ ] Graceful shutdown: drain in-flight requests before scaling down
- [ ] Health checks: model warmup endpoint (load weights before accepting traffic)
- [ ] Circuit breaker: if error rate > 10% in 1 min, fall back to secondary model
- [ ] Cost monitoring: per-request cost tracking (model + tokens + infra)

---

## Summary

Choose self-hosting for control and privacy; choose managed for speed and flexibility. Scale horizontally with queues, protect with rate limiting, and save money with semantic caching. Route requests to the right-sized model for the job. Always deploy with health checks, canaries, and circuit breakers.

---

## Key Terms
- **PagedAttention:** Memory management technique in vLLM that reduces GPU memory waste
- **Continuous batching:** Adding new requests to a running batch as others complete
- **Semantic cache:** A vector-similarity cache that returns results for semantically similar queries
- **Circuit breaker:** A pattern that stops calling a failing service to prevent cascading failures
