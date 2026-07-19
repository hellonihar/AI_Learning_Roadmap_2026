# 07.3.1 — Efficient Attention

## Overview

The standard attention mechanism in Transformers scales **quadratically** with sequence length:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

This $O(L^2)$ cost becomes prohibitive for long sequences (e.g., 100K+ tokens). Efficient attention mechanisms reduce this complexity to $O(L)$ or $O(L \log L)$ while preserving modelling power.

## Sparse Attention

Replace the full $L \times L$ attention matrix with a sparse pattern. Common patterns:

- **Fixed patterns**: each token attends to local neighbours (sliding window) and a set of global tokens.
- **Strided patterns**: alternate between attending to nearby tokens and attending to every $k$-th token.
- **BP-Transformer**: binary partitioning tree for hierarchical attention.

The **Generating Long Sequences with Sparse Transformers** (Child et al., 2019) paper showed that sparse attention enables modelling sequences of length 64K+.

## Sliding Window Attention

Each token attends only to $W$ neighbours on each side (window size $W$). Complexity drops to $O(L \cdot W)$. Used in Mistral and GPT-4's long context:

$$
\text{Attention}_{ij} = 0 \quad \text{if } |i - j| > W
$$

Sliding window attention can be stacked: earlier layers capture local patterns, deeper layers capture global structure. With $k$ layers of window size $W$, each token indirectly attends to $k \cdot W$ tokens.

## FlashAttention

**Paper**: *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*

FlashAttention (Dao et al., 2022) is an **exact** attention algorithm that is faster and more memory-efficient by being IO-aware. Instead of materialising the full $L \times L$ attention matrix in HBM (high-bandwidth memory), it computes attention in tiles using SRAM:

1. **Tiling**: load blocks of $Q, K, V$ from HBM to SRAM.
2. **Online softmax**: compute attention incrementally without materialising the full matrix.
3. **Recomputation**: recompute attention during backward pass (avoids storing $L \times L$ matrix).

FlashAttention achieves 2-4x speedup and reduces memory from $O(L^2)$ to $O(L)$. FlashAttention-2 further optimises the algorithm for improved parallelism.

## Multi-Query and Grouped-Query Attention

In standard multi-head attention, each head has separate $K$ and $V$ projections. **Multi-query attention** (MQA) shares a single $K$ and $V$ across all heads, dramatically reducing memory during autoregressive decoding. **Grouped-query attention** (GQA) is a compromise: heads are divided into $g$ groups, each sharing $K$ and $V$.

$$
\text{MQA: } \text{heads} \times Q, \quad 1 \times K, \quad 1 \times V
$$
$$
\text{GQA: } \text{heads} \times Q, \quad g \times K, \quad g \times V
$$

GQA with 8 groups (used in Llama 2 70B) nearly matches full multi-head quality while being significantly faster at inference time.

## Practical Context

Efficient attention is essential for long-context models (Claude 200K, GPT-4 128K). FlashAttention is the default implementation in Hugging Face Transformers and几乎所有 modern LLM inference frameworks.
