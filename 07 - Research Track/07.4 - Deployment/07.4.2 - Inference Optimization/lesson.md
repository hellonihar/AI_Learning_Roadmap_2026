# 07.4.2 — Inference Optimization

## Overview

Inference optimisation encompasses all techniques to reduce latency and increase throughput of LLM inference without changing the output distribution. Optimisations span the entire stack: from low-level kernel fusion to high-level parallelism strategies.

## Kernel Fusion

LLM inference is composed of many small GPU kernel launches (matrix multiplies, softmax, layer norm, elementwise ops). Each kernel launch has overhead. **Kernel fusion** combines multiple operations into a single kernel, reducing launch overhead and improving data locality.

**Fused operations** in practice:

- **Fused attention**: combine $QK^T$, softmax, dropout, $AV$ in one kernel (FlashAttention does this within a tile).
- **Fused MLP**: combine the two linear layers and activation (e.g., SiLU) in a single kernel.
- **Fused layer norm**: fuse normalisation, scale, and bias into one kernel.

NVIDIA TensorRT-LLM and xFormers provide extensive kernel fusion. For example, the **fused multi-head attention** kernel in xFormers is ~40% faster than separate kernel launches.

## Memory Management

LLM inference is **memory-bound** in the decode phase — the bottleneck is moving weights/KV cache from HBM to compute units, not the computation itself.

**Memory bandwidth is the constraint**: an A100-80GB has ~2 TB/s memory bandwidth. Reading all 140GB of a 70B model in FP16 takes ~70ms — this sets a floor on per-token latency.

Optimisation strategies:

- **Quantisation** (see 07.3.3): INT4 reduces weight memory by 4x.
- **KV cache optimisation** (see 07.3.2): PagedAttention, KV cache quantisation, shared prefix.
- **Prefetching**: overlap memory loads with computation.
- **Memory pooling**: reuse GPU memory allocations to avoid fragmentation.

## Tensor Parallelism (TP)

Shard each layer's weights across GPUs. For a $d_{\text{model}} \times d_{\text{ff}}$ linear layer with 2 GPUs:

- GPU 0: weights $[:, :d_{\text{ff}}/2]$
- GPU 1: weights $[:, d_{\text{ff}}/2:]$

After computation, an **all-reduce** operation combines results. TP requires high-bandwidth GPU interconnects (NVLink ~600 GB/s) and adds communication overhead. Typical TP degree: 1-8 GPUs.

## Pipeline Parallelism (PP)

Split the model by layers across GPUs. GPU 0 handles layers 0-15, GPU 1 handles layers 16-31, etc. During inference:

- **Prefill**: micro-batches flow through the pipeline.
- **Decode**: each token passes through all pipeline stages.

The key challenge is **bubble** (idle time when micro-batches propagate). For $m$ micro-batches and $p$ pipeline stages, bubble overhead is $\frac{p-1}{m}$.

**1F1B** (one-forward-one-backward) scheduling minimises bubble. For pure inference (no backward pass), PP is simpler than for training but still introduces latency from inter-GPU communication.

## Model Parallelism Comparison

| Type | Granularity | Comm. | Best for |
|------|-------------|-------|----------|
| **Tensor (TP)** | Per layer | All-reduce | Single-node (NVLink) |
| **Pipeline (PP)** | Layer groups | P2P | Multi-node |
| **Sequence (SP)** | Sequence chunks | All-to-all | Long contexts |

Many serving systems combine TP + PP for large models (e.g., TP=8, PP=2 for 16 GPUs).

## Practical Context

In practice, inference optimisation is about reducing **time-to-first-token (TTFT)** for the prefill phase and **tokens-per-second (TPS)** for the decode phase. TensorRT-LLM and vLLM incorporate most of these optimisations, making them the recommended deployment frameworks for latency-sensitive applications.
