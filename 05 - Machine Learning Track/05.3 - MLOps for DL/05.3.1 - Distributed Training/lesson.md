# Lesson 05.3.1: Distributed Training

## Overview

Modern deep learning models — from GPT-4 with trillions of parameters to vision transformers with billions — far exceed the memory capacity of a single GPU. Distributed training is the practice of spreading computation across multiple accelerators to reduce training time and enable model scales that would otherwise be impossible.

This lesson covers the core parallelism strategies, the ZeRO optimization family, and the communication primitives that make distributed training work.

---

## 1. Data Parallelism

Data parallelism is the simplest and most widely adopted form of distributed training.

**How it works:** Each GPU holds a complete copy of the model. The global batch is split into micro-batches, one per GPU. Each GPU computes forward and backward passes independently. Gradients are then averaged across all GPUs (via all-reduce), and every GPU applies the identical update.

**Key characteristics:**
- Model must fit on a single GPU.
- Scales well up to hundreds of GPUs for moderate-size models.
- Communication overhead grows with model size (all-reduce of full gradient tensors).

**Pseudocode:**
```
for each batch:
    split batch into N micro-batches
    each GPU: forward(micro_batch) → loss
    each GPU: backward(loss) → gradients
    all-reduce(gradients)           # average across GPUs
    each GPU: optimizer.step()
```

**Framework support:** `torch.nn.DataParallel` (obsolete), `torch.nn.DistributedDataParallel` (DDP, preferred), `tf.distribute.MirroredStrategy`.

---

## 2. Model Parallelism

When the model is too large for a single GPU, model parallelism splits the model itself across devices.

### 2.1 Naive Model Parallelism

Layers are partitioned across GPUs. GPU 0 holds layers 1–3, GPU 1 holds layers 4–6, and so on. A single micro-batch passes through GPU 0, the intermediate activations are transferred to GPU 1, etc.

**Downside:** At any moment only one GPU is active — severe GPU under-utilisation (no compute/memory overlap).

### 2.2 Pipeline Parallelism

Pipeline parallelism improves on naive model parallelism by introducing micro-batching within the macro-batch.

GPUs are arranged in a pipeline. The input batch is split into micro-batches that are fed sequentially. While GPU 0 processes micro-batch `i+1`, GPU 1 processes the output of micro-batch `i`. This keeps all GPUs busy after the pipeline fill stage.

**Popular implementations:**
- **GPipe**: Divides micro-batches evenly; uses gradient accumulation across micro-batches.
- **PipeDream**: Uses 1F1B (one-forward-one-back) scheduling for better memory efficiency.

**Trade-off:** Pipeline bubbles (idle time at start and end) reduce efficiency. Deeper pipelines with too few micro-batches suffer more.

### 2.3 Tensor Parallelism

Tensor parallelism splits individual operations (e.g., matrix multiplies) across GPUs. A single layer's weight matrix `W` is sharded column-wise or row-wise.

For a linear layer `Y = X @ W`:
- **Column-wise sharding:** Split `W = [W1, W2]` across GPUs. Each GPU computes `X @ Wi`. An all-reduce concatenates results.
- **Row-wise sharding:** Split `W = [W1; W2]`. Each GPU computes `X_i @ Wi`, and all-gather combines outputs.

Tensor parallelism requires high-bandwidth intra-node communication (NVLink, NVSwitch) because every forward/backward call triggers collective operations. It is a key component of **Megatron-LM** and **Megatron-DeepSpeed**.

---

## 3. FSDP: Fully Sharded Data Parallel

FSDP, developed by Facebook AI Research, is a hybrid strategy that marries data parallelism's ease-of-use with model parallelism's memory efficiency.

**Core idea:** During forward/backward, each FSDP unit (typically a layer) is all-gathered so every GPU has the full parameters. After the unit's computation, parameters are freed (resharded). Only the currently needed layer is materialised on each GPU at any time.

**Memory savings:**
- DDP: `O(model_size)` per GPU.
- FSDP: `O(model_size / num_gpus)` per GPU for parameters + activations.

**FSDP sharding strategies:**
- `SHARD_GRAD_OP`: Only shard gradients (ZeRO stage 2 equivalent).
- `FULL_SHARD`: Shard parameters, gradients, and optimizer states (ZeRO stage 3 equivalent).
- `NO_SHARD`: Equivalent to DDP.

**Usage (PyTorch):**
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=transformer_auto_wrap_policy,
)
```

FSDP is the recommended starting point for training models in the 1B–175B parameter range on PyTorch.

---

## 4. DeepSpeed and ZeRO

DeepSpeed is a Microsoft library that implements the **ZeRO (Zero Redundancy Optimizer)** family of optimisations.

### ZeRO Stages

| Stage | What is sharded | Memory reduction vs DDP | Communication |
|-------|----------------|------------------------|---------------|
| **ZeRO-1** | Optimizer states only | 4× | Same as DDP |
| **ZeRO-2** | Optimizer states + gradients | 8× | Same as DDP |
| **ZeRO-3** | Optimizer states + gradients + parameters | `N`× (linear with GPU count) | Increased (parameters all-gathered each step) |

- **ZeRO-1/2** add no extra communication over DDP; all-reduce is replaced by reduce-scatter + all-gather but the total volume is identical.
- **ZeRO-3** incurs more communication because parameters must be all-gathered before each forward/backward segment.

**ZeRO-Offload:** Offloads optimizer states and gradients to CPU/NVMe when GPU memory is exhausted. Trade-off: slower step time due to CPU<->GPU transfers.

**ZeRO-Infinity:** Extends offloading to parameters as well; enables training of trillion-parameter models on limited GPU clusters.

**DeepSpeed configuration (JSON):**
```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 4,
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu" }
  }
}
```

---

## 5. Communication Primitives

Distributed training relies on **collective communication** operations, typically implemented via **NCCL** (NVIDIA Collective Communication Library) on NVIDIA GPUs, or **RCCL** on AMD GPUs.

### Key Collectives

| Operation | Description | Used in |
|-----------|-------------|---------|
| **All-reduce** | Sum/average values across all ranks; every rank gets the result. | DDP gradient averaging |
| **Reduce-scatter** | Sum values, then scatter chunks to each rank. | ZeRO-2/3 gradient reduction |
| **All-gather** | Gather full set of values from all ranks onto every rank. | ZeRO-3 parameter materialisation |
| **Broadcast** | Send one rank's value to all others. | Initial parameter synchronisation |
| **P2P (send/recv)** | Point-to-point transfer between two ranks. | Pipeline parallelism |

### NCCL Optimisations

- **NCCL ring algorithm:** Efficient all-reduce for large tensors (splits data into chunks passed around a ring).
- **NCCL tree algorithm:** Lower latency for small tensors (used as fallback).
- **NCCL_MIN_NCHANNELS, NCCL_ALGO, NCCL_PROTO:** Environment variables to tune bandwidth utilisation.
- **NVLink/NVSwitch:** High-bandwidth (600 GB/s+) GPU-GPU interconnect within a node.

### Bandwidth Considerations

- Intra-node (NVLink): ~600 GB/s.
- Inter-node (InfiniBand): ~50–400 Gb/s (6.25–50 GB/s).
- Inter-node (Ethernet): ~25–100 Gb/s (3.125–12.5 GB/s).

The slowest link in the collective communication path determines overall scaling efficiency. Gradient compression (e.g., PowerSGD, 1-bit SGD) can reduce bandwidth pressure.

---

## 6. Choosing a Strategy

| Model size | Recommended strategy |
|------------|---------------------|
| < 1B params | DDP or FSDP (NO_SHARD) |
| 1B–10B params | FSDP (FULL_SHARD) or DeepSpeed ZeRO-2/3 |
| 10B–100B params | DeepSpeed ZeRO-3 + Pipeline Parallelism |
| 100B–1T+ params | 3D Parallelism (Data + Pipeline + Tensor) |

**3D Parallelism** (Megatron-DeepSpeed) combines all three: tensor parallelism within a node (NVLink), pipeline parallelism across nodes, and data parallelism across pipeline replicas.

---

## Summary

- **Data parallelism:** Simple, requires model to fit on one GPU.
- **Pipeline parallelism:** Improves utilisation for large models; suffers from bubbles.
- **Tensor parallelism:** Splits individual operations; needs high-bandwidth intra-node links.
- **FSDP:** Memory-efficient data parallelism; shards parameters/gradients/optimiser states.
- **DeepSpeed/ZeRO:** Industry standard for large-scale training; stages trade memory for communication.
- **NCCL:** The low-level communication library; choice of algorithm impacts throughput.

Next lesson: **05.3.2 — Checkpointing**, where we cover saving and resuming training state in distributed environments.
