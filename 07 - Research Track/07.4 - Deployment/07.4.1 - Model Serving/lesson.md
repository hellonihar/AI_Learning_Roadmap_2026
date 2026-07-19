# 07.4.1 — Model Serving

## Overview

Model serving systems bridge the gap between trained models and production users. For LLMs, serving is uniquely challenging because of the autoregressive decoding loop, memory-bound attention, and the need to handle thousands of concurrent requests with low latency.

## vLLM

**Paper**: *Efficient Memory Management for Large Language Model Serving with PagedAttention*

vLLM is the most popular open-source LLM serving engine. Its key innovations:

- **PagedAttention**: manages KV cache in fixed-size blocks, eliminating memory fragmentation and enabling near-100% cache utilisation.
- **Continuous batching**: dynamically adds/removes sequences from a batch as they complete generation, maximising GPU utilisation.
- **Speculative decoding**: integrated support for draft-model verification.
- **Tensor parallelism**: shard model across multiple GPUs for large models.

vLLM supports most popular models (Llama, Mistral, Falcon, GPT-NeoX) and achieves 2-4x throughput improvement over naive implementations. It exposes an OpenAI-compatible API via `vllm serve`.

## TGI — Text Generation Inference

Hugging Face's TGI is a production-grade serving solution designed for Hugging Face models. Features:

- **Continuous batching** (adopted from the ORCA paper)
- **Tensor parallelism** via sharding
- **Safetensors loading** for fast startup
- **Watermarking and content moderation** built-in
- **FlashAttention** integration
- **Speculative decoding** (assisted generation)

TGI uses **server-sent events (SSE)** for streaming and provides Prometheus metrics out of the box.

## Triton Inference Server

NVIDIA Triton is a general-purpose inference server supporting multiple frameworks (PyTorch, TensorRT, ONNX). For LLMs, TensorRT-LLM backend provides:

- **In-flight batching**: dynamic batching with prioritisation
- **Paged KV cache**: similar to vLLM
- **INT4/INT8 quantisation**: using TensorRT optimisations
- **Multi-GPU multi-node serving**: pipeline + tensor parallelism

Triton excels in environments requiring multi-framework support, advanced request scheduling, and GPU sharing across models.

## Continuous Batching

**Paper**: *ORCA: A Distributed Serving System for Transformer-Based Generative Models*

Continuous batching (also called **in-flight batching**) dynamically manages the generation batch. After each step, completed sequences are removed and new sequences (or prefill requests) are added. This contrasts with **static batching**, where all sequences start and end together.

Continuous batching is the single most important throughput optimisation for LLM serving. It keeps the GPU fully utilised even with varying sequence lengths.

## Tensor Parallelism for Serving

Large models (70B+) cannot fit on a single GPU. Tensor parallelism shards each layer's weights across multiple GPUs. For attention:

$$
Q = [Q_1, Q_2, \ldots, Q_g], \quad K = [K_1, \ldots, K_g], \quad V = [V_1, \ldots, V_g]
$$

Each GPU computes attention on its shard, then all-gathers the results. This requires fast interconnect (NVLink, InfiniBand). For 8 GPUs serving a 70B model, tensor parallelism over 8-way is standard.

## Practical Context

For most teams, vLLM is the default choice for open-source model serving. TGI is preferred when running Hugging Face models with minimal code changes. Triton is chosen for enterprise deployments requiring multi-framework support and advanced scheduling.
