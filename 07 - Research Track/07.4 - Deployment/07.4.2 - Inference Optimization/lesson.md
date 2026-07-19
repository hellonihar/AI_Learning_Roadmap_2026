# 07.4.2 — Inference Optimization

## Overview

Inference optimisation encompasses all techniques to reduce latency and increase throughput of LLM inference without changing the output distribution. Optimisations span the entire stack: from low-level kernel fusion to high-level parallelism strategies.

## Kernel Fusion

LLM inference is composed of many small GPU kernel launches (matrix multiplies, softmax, layer norm, elementwise ops). Each kernel launch has overhead — the CPU dispatches work to the GPU via a command queue, and the GPU must read kernel instructions, load data from HBM, write results back, and synchronise. **Kernel fusion** combines multiple operations into a single GPU kernel, reducing launch overhead and improving data locality by keeping intermediate values in on-chip SRAM (shared memory/registers) instead of writing to HBM.

### Why Fusion Matters

For a typical transformer layer during inference, the operation graph looks like:

```
Input → LayerNorm → QKV投影 → Attention (QK^T → Softmax → AV) → 
Output projection → Residual add → LayerNorm → MLP up → Activation → 
MLP gate → MLP down → Residual add → Output
```

Without fusion, this is **15+ separate kernel launches per layer per token**. With a 70B model having 80 layers and generating 1024 tokens, that's **~1.2 million kernel launches** for a single sequence. Each launch has ~5-50μs overhead on CUDA, wasting hundreds of milliseconds purely on scheduling.

### Types of Kernel Fusion

#### 1. Pointwise Fusion (Elementwise Ops)

The simplest fusion: combine consecutive elementwise operations that have the same launch grid dimensions.

**Example — Layer Normalisation** (unfused):
```
kernel 1: compute mean        (launch: read x, write mean)
kernel 2: compute variance    (launch: read x, mean, write var)
kernel 3: normalise           (launch: read x, mean, var, γ, β, write y)
```

**Fused LayerNorm** combines all three into one kernel:
```cuda
__global__ void fused_layer_norm(float* x, float* y, float* gamma, float* beta, int N) {
    // Single pass: compute mean, variance, and normalise in shared memory
    // Intermediate values stay in registers/SRAM — no HBM roundtrips
}
```

**Performance gain:** ~1.5-2× vs three separate launches for LayerNorm.

#### 2. Horizontal Fusion (Same Input, Different Ops)

Combine operations that read the same input but produce different outputs — common in attention's QKV projection.

**Unfused QKV projection:**
```
kernel 1: x @ W_Q  → Q    (read x, write Q)
kernel 2: x @ W_K  → K    (read x again from HBM, write K)
kernel 3: x @ W_V  → V    (read x again from HBM, write V)
```

**Fused QKV projection:** One kernel that loads `x` once into registers/shared memory, computes all three matmuls, and writes Q, K, V contiguously:
```cuda
__global__ void fused_qkv(float* x, float* W_qkv, float* Q, float* K, float* V) {
    // Load x once into shared memory
    // Compute x @ W_Q, x @ W_K, x @ W_V in a single pass
    // Write Q, K, V as contiguous blocks in one go
}
```

**Performance gain:** ~1.7× by eliminating redundant HBM reads of the input `x`. Also improves memory layout — Q, K, V can be written as a single contiguous block, enabling better cache behaviour for the next operation.

#### 3. Vertical Fusion (Producer-Consumer Chains)

The most impactful type: fuse a sequence of operations where each step's output is immediately consumed by the next. This keeps intermediate data in on-chip memory, avoiding HBM roundtrips entirely.

**Fused Attention (FlashAttention):** The canonical example. Instead of writing the full attention matrix $S = QK^T \in \mathbb{R}^{N \times N}$ to HBM (which costs $O(N^2)$ memory for long sequences), FlashAttention tiles the computation:

```python
# Pseudo-code for fused attention
for block_Q in tiles(Q):
    for block_K, block_V in tiles(K, V):
        # Load tiles into SRAM (fast, ~20-40x cheaper than HBM)
        s_block = block_Q @ block_K.T    # Compute in registers
        p_block = softmax(s_block)        # Online softmax with rescaling
        o_block += p_block @ block_V      # Accumulate output
    # Write final O block to HBM once
```

**What's fused:** $QK^T$ → Softmax → Dropout → $AV$ — all within a single kernel scope, processing in tiles. The attention matrix $S$ is **never materialised in HBM**.

**Performance gain:** FlashAttention achieves **2-4× end-to-end speedup** over unfused attention for long sequences, and uses $O(N)$ memory instead of $O(N^2)$.

**Fused MLP (Gated variants like SwiGLU):** The MLP in modern LLMs (LLaMA, GPT-J) uses a gated structure:

```
Unfused:          Fused:
x @ W_gate → gate   │
x @ W_up   → up     ├── fused_gated_mlp(x):
gate * SiLU(up) → act│    x @ [W_gate | W_up]  (one matmul)
act @ W_down → out  │    act = gate * SiLU(up)  (elementwise fuse)
                    │    act @ W_down
```

The fused kernel concatenates `W_gate` and `W_up` into a single matrix, performs one large matmul instead of two smaller ones (better GPU utilisation), and applies the activation in the same kernel scope.

**Performance gain:** ~1.3-1.5× for the MLP block.

**Fused Residual + LayerNorm (Pre-norm):**

```
Unfused:                           Fused:
x' = x + attention(x)  ──┐        │
ln(x') → x_norm         ├──┐      ├── fused_residual_ln(x, attn_out):
                         │  │      │   x' = x + attn_out    (elementwise)
                         │  │      │   mean = mean(x')
                         │  │      │   var  = var(x')
                         │  │      │   x_norm = (x' - mean) / sqrt(var + eps) * γ + β
                         │  │      │   return x', x_norm
```

**Performance gain:** ~1.5× by keeping `x'` in registers — no HBM write for the residual output.

### GPU Hardware & Fusion

Kernel fusion effectiveness depends on GPU architecture:

| Architecture | SRAM/shared memory per SM | Ideal fusion target |
|---|---|---|
| V100 (Volta) | 96 KB | Elementwise + small reductions |
| A100 (Ampere) | 192 KB | Attention tiles up to 64×64 |
| H100 (Hopper) | 228 KB | Larger tiles, FP8 tensor core fusion |
| H200 | 228 KB + faster HBM3e | Same as H100 |

**Constraint:** Fused kernels are limited by register pressure and shared memory size. If a fused kernel uses too many registers, the GPU reduces occupancy (fewer warps per SM), potentially negating fusion benefits. This is the key tradeoff — fusion reduces HBM traffic but may reduce parallelism.

### Practical Fusion in Frameworks

#### NVIDIA TensorRT-LLM
- **Graph-level fusion:** TensorRT analyses the entire model graph and identifies fusion opportunities automatically. It fuses LayerNorm, residual adds, QKV projections, and MLP blocks.
- **Plugins:** Custom fused kernels for attention (FlashAttention-2), gated MLP, and RoPE (rotary position embedding).
- **Result:** Single transformer layer goes from ~15 kernels to ~3-4 fused kernels.

```python
# TensorRT-LLM automatically fuses this during build phase
model = LLaMAForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
# Build engine — TRT analyses and fuses ops automatically
builder = trt.Builder(logger)
config = builder.create_builder_config()
config.plugin_config.set_attention(AttentionType.FLASH_ATTENTION)
engine = builder.build_serialized_network(network, config)
```

#### xFormers (Meta)
- Provides drop-in fused attention kernels (`memory_efficient_attention`).
- Fused softmax + masking + dropout in a single kernel.
- Supports arbitrary attention masks without materialising them.

```python
from xformers.ops import memory_efficient_attention

# Single fused kernel call — internally does QK^T, softmax, masking, AV
output = memory_efficient_attention(q, k, v, attn_bias=attention_mask)
```

#### vLLM (PagedAttention)
- Fuses attention with its PagedAttention memory manager.
- The attention kernel directly reads/writes KV-cache blocks in page tables, avoiding separate memory management overhead.

#### OpenAI Triton (DSL)
- A Python-based DSL for writing custom fused kernels.
- Lets you write fused operators without hand-tuning CUDA:

```python
import triton
import triton.language as tl

@triton.jit
def fused_attention_kernel(Q, K, V, O, ...):
    # Triton compiler auto-fuses tile loads, matmul, softmax, writes
    # into a single optimised CUDA kernel
```

### Measuring Fusion Impact

| Metric | Unfused | Fused | Improvement |
|--------|---------|-------|-------------|
| Kernels per transformer layer | 15 | 3-4 | 4-5× fewer launches |
| HBM reads per layer (7B, FP16) | ~3.2 GB | ~1.8 GB | 44% reduction |
| Latency per token (7B, A100) | ~8 ms | ~5 ms | 37% reduction |
| Throughput (tokens/sec, A100) | ~2,800 | ~4,500 | 60% improvement |

### Key Takeaways

- Kernel fusion is the single most impactful GPU-level optimisation for inference — it reduces launch overhead and HBM traffic simultaneously.
- The most important fusions for LLMs are: QKV projection, attention (FlashAttention), gated MLP, and residual + layer norm.
- Fusion effectiveness is limited by GPU shared memory — the tradeoff is HBM savings vs. register pressure.
- Modern frameworks (TensorRT-LLM, xFormers, vLLM) perform fusion automatically; understanding fusion helps you choose the right framework and diagnose performance bottlenecks.

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
