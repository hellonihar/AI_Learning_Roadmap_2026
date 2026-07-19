# 07.3.2 — KV Cache

## Overview

Autoregressive decoding generates one token at a time. At each step, the model recomputes attention over all previous tokens. The **KV cache** eliminates redundant computation by caching the Key and Value tensors from previous steps, trading memory for computation.

## The Caching Mechanism

During generation of token $t$, the model computes:

$$Q_t = x_t W_Q, \quad K_t = x_t W_K, \quad V_t = x_t W_V$$

Without caching, attention for step $t$ requires recomputing all previous $(K_{<t}, V_{<t})$ from scratch — $O(t^2)$ total cost over $t$ steps. With KV cache, we store $(K_{<t}, V_{<t})$ in memory. For step $t$:

1. Compute $Q_t, K_t, V_t$ for the current token.
2. Append $K_t, V_t$ to the cache.
3. Compute attention:

$$
\text{Attention}_t = \text{softmax}\left( \frac{Q_t [K_{<t}; K_t]^T}{\sqrt{d_k}} \right) [V_{<t}; V_t]
$$

The per-step cost is $O(t)$ instead of $O(t^2)$, a **linear** reduction.

## Memory Challenges

The KV cache is the primary memory bottleneck during long-context inference:

$$
\text{Memory per token} = 2 \times (\text{layers}) \times (\text{heads}) \times (\text{head dim}) \times \text{sizeof(dtype)}
$$

For a 70B model with 80 layers, 8 KV-heads, 128 head-dim, in FP16:

$$
80 \times 8 \times 128 \times 2 \text{ bytes} = 163,840 \text{ bytes per token}
$$

For a 32K context: $32K \times 163,840 \approx 5\text{GB}$ of cache — per request.

## PagedAttention and vLLM

**Paper**: *Efficient Memory Management for Large Language Model Serving with PagedAttention*

KV cache memory suffers from **fragmentation** (internal and external). PagedAttention manages the KV cache in fixed-size **blocks** (pages), analogous to virtual memory in operating systems:

- **Non-contiguous physical pages**: blocks need not be contiguous in memory.
- **Copy-on-write**: multiple requests sharing a prompt prefix share physical pages.
- **Demand paging**: pages are allocated on demand, reducing waste.

vLLM implements PagedAttention and achieves near-zero waste, improving throughput by 2-4x compared to naive caching.

## Other Optimisations

- **KV cache quantisation**: store cache in INT4/INT8 (e.g., KIVI, KVQuant).
- **Windowed cache**: only keep the last $W$ tokens (sliding window), discard older ones.
- **Shared prefix cache**: reuse KV cache across requests with the same system prompt.
- **Key-value offloading**: offload cache to CPU when idle, prefetch before use.

## Practical Context

KV cache management is the single most important optimisation for LLM serving. vLLM has become the standard inference engine largely due to its PagedAttention implementation. Understanding KV cache is essential for anyone deploying LLMs at scale.
