# 07.4.3 — Cost Optimization

## Overview

LLM inference is expensive. A 70B model running on 8×A100s costs ~$30/hour. Cost optimisation strategies reduce this by improving hardware utilisation, using cheaper hardware, and minimising unnecessary computation — without significantly degrading user experience.

## Spot Instances

Cloud GPU providers offer **spot (preemptible) instances** at 60-90% discount compared to on-demand pricing. The trade-off: instances can be reclaimed with 2-minute notice.

For LLM serving:

- **Stateless inference**: if a request is interrupted, the client retries. This works well with load balancers that redirect requests.
- **Graceful degradation**: use a mix of on-demand (reliable) and spot (cheap) instances. Under normal load, spot handles most traffic; during pre-emption, requests drain to on-demand.
- **Model replicas**: run multiple model replicas across spot instances. If one is reclaimed, others handle the load.

Tools like **Karpenter** (Kubernetes) and **SkyPilot** automate spot instance management.

## Model Routing (Small vs Large)

Not all requests need a 70B model. **Model routing** sends simple queries to a small, cheap model and complex queries to a large model.

**Router types**:

- **Heuristic**: route based on query length, domain, or keyword patterns. Example: "What is the capital of France?" → small model; "Explain the proof of Fermat's Last Theorem" → large model.
- **Classifier-based**: train a lightweight classifier (e.g., DistilBERT) to predict which model is sufficient.
- **Proxy-score**: use a small model's confidence score as a routing signal. If confidence > threshold, return the answer; otherwise, escalate to the large model.

**Savings**: 50-80% cost reduction while maintaining 95%+ of quality, depending on the routing strategy and query distribution.

## Caching

LLM inference is dominated by repeated queries (system prompts, common questions, few-shot examples). **Response caching** avoids redundant computation.

**Cache types**:

- **Exact match**: identical prompt → return cached response. Simple, high hit rate for system prompts.
- **Semantic caching**: retrieve similar prompts via embedding similarity. Use a vector database for nearest-neighbour lookup.
- **Prefix caching**: cache KV cache for common prefixes (e.g., system prompt). vLLL's automatic prefix caching shares KV cache across requests with the same prefix.

Cache hit rates of 30-60% are common in production, depending on the application. Semantic caching with embedding similarity (cosine > 0.95) adds ~5ms of overhead but can increase hit rate significantly.

## Batching Strategy

**Dynamic batching**: queue requests and process them together. Larger batches amortise the fixed cost of weight loading and improve GPU utilisation.

**Trade-off**: latency vs throughput. Small batches → low latency, low throughput. Large batches → high throughput, higher latency (queueing + longer iteration time).

For LLMs, **continuous batching** (see 07.4.1) is optimal: new requests join an in-flight batch as sequences complete, maintaining high utilisation without fixed wait times.

**Optimal batch size** depends on model size, GPU memory, and latency targets. A good starting point: fill GPU memory to 80-90% with the KV cache.

## GPU Selection

Different GPUs offer different cost-performance trade-offs:

| GPU | Memory | HBM BW | Relative Cost | Best for |
|-----|--------|--------|--------------|----------|
| A100-80GB | 80 GB | 2 TB/s | 1x | Large models, high throughput |
| A100-40GB | 40 GB | 1.6 TB/s | 0.7x | 7B-13B models |
| L40S | 48 GB | 864 GB/s | 0.4x | Mid-range, good value |
| H100 | 80 GB | 3.35 TB/s | 1.5x | Premium performance |
| T4 | 16 GB | 320 GB/s | 0.1x | Small models, edge |

**Guideline**: for a given model, choose the cheapest GPU that fits the model (with quantisation) and meets your latency target. A quantised 7B model runs well on a T4; a 70B model needs at least an A100-80GB (or multiple A100-40GB with TP).

## Practical Context

Cost optimisation is an ongoing process: start with on-demand GPUs, add caching, then introduce model routing, then switch to spot instances for some traffic. Monitor time-per-output-token and cost-per-request. A well-optimised deployment can reduce costs by 5-10x compared to a naive setup.
