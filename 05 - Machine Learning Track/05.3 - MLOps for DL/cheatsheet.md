# Cheatsheet: MLOps for Deep Learning

## 1. Distributed Training Strategies

| Strategy | Model fits on 1 GPU? | Comm overhead | Memory per GPU | When to use |
|----------|---------------------|---------------|----------------|-------------|
| **DDP** | Yes | Low (all-reduce gradients) | O(model) | Models < 1B params |
| **FSDP (FULL_SHARD)** | Not required | Medium (all-gather params) | O(model / GPUs) | 1B–10B params |
| **DeepSpeed ZeRO-2** | Not required | Low (same as DDP) | O(model / GPUs) + opt states | 1B–10B params |
| **DeepSpeed ZeRO-3** | Not required | High (all-gather each step) | O(model / GPUs) | 10B–100B params |
| **Pipeline Parallel** | No | P2P transfers | O(segment) | Models > 10B |
| **Tensor Parallel** | No | All-reduce per op | O(model / GPUs) | Intra-node > 10B |
| **3D Parallelism** | No | All types | Minimal | > 100B params |

### Collective Communication Ops

| Op | Description | Used by |
|----|-------------|---------|
| `all_reduce` | Sum across ranks, result to all | DDP gradient sync |
| `reduce_scatter` | Sum, then scatter pieces | ZeRO-2/3 gradients |
| `all_gather` | Gather full tensor on each rank | ZeRO-3 params, FSDP |
| `broadcast` | One rank → all others | Init, seed sync |
| `send`/`recv` | P2P between two ranks | Pipeline parallel |

### DeepSpeed ZeRO Comparison

```
ZeRO-1:   Shard optimizer states    → 4× memory reduction vs DDP
ZeRO-2:   Shard optimizer + grads   → 8× memory reduction vs DDP
ZeRO-3:   Shard optimizer + grads + params → N× (N = GPU count)
ZeRO-Offload: Offload optimizer states to CPU/NVMe
ZeRO-Infinity: Offload everything (params + opt + grads)
```

---

## 2. Checkpointing

### What to Save

```
Checkpoint = {
    "model_state_dict": ...,      # Required
    "optimizer_state_dict": ...,  # Required for resume
    "scheduler_state_dict": ...,  # Required for resume
    "epoch": int,                 # Required for resume
    "global_step": int,           # Required for resume
    "loss": float,                # Optional (logging)
    "rng_state": ...,             # Optional (reproducibility)
    "grad_scaler": ...,           # Required if AMP
}
```

### Resume Logic

```python
if resume_from:
    checkpoint = torch.load(resume_from, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
else:
    start_epoch = 0
```

### FSDP Checkpointing

| Mode | Memory | Disk | Steps |
|------|--------|------|-------|
| FULL_STATE_DICT | High (rank 0) | 1 file | `with FSDP.state_dict_type(... FULL_STATE_DICT ...)` |
| SHARDED_STATE_DICT | Low | N files | `dcp.save(state_dict, ...)` |

### Checkpoint Frequency

- **Periodic:** Every 30–60 min of training.
- **Best-model:** On validation metric improvement.
- **Final:** At end of training (+ signal handler for spot preemption).

---

## 3. Weight Formats & Storage

### Numerical Formats

| Format | Bits | Use case |
|--------|------|----------|
| FP32 | 32 | Master weights, optimizer states |
| FP16 | 16 | Forward/backward, inference |
| BF16 | 16 | Training (same range as FP32) |
| INT8 | 8 | Inference (CPU/GPU) |
| INT4 | 4 | LLM inference (quantized) |
| FP8 | 8 | H100 training/inference |

### Quantization Types

| Method | Training needed? | Accuracy | Bits | Tools |
|--------|-----------------|----------|------|-------|
| RTN (round-to-nearest) | No | Low | 4/8 | bitsandbytes |
| GPTQ | No (calibration) | Medium | 2/3/4/8 | AutoGPTQ |
| AWQ | No (calibration) | High | 4 | AutoAWQ |
| QAT | Yes | Highest | 2/4/8 | torch.ao, TensorRT |
| NF4 (QLoRA) | No | High | 4 | bitsandbytes |

### Storage Comparison

```
1B params FP32 = 4.0 GB
1B params FP16 = 2.0 GB
1B params INT8 = 1.0 GB
1B params INT4 = 0.5 GB
70B params FP16 = 140 GB
70B params INT4 = 35 GB
```

### SafeTensors vs Pickle

```
                 Pickle      SafeTensors
Security         Unsafe      Safe
Speed            Slow        Fast (mmap)
Lazy loading     No          Yes
Hub default      Legacy      Current
```

---

## 4. Inference Optimisation

### Engine Comparison

| Engine | GPU | CPU | LLM opt | Quantization |
|--------|-----|-----|---------|-------------|
| PyTorch eager | ✓ | ✓ | ✗ | torch.ao |
| TorchScript | ✓ | ✓ | ✗ | Via torch.ao |
| ONNX Runtime | ✓ | ✓ | ✗ | Built-in INT8 |
| TensorRT | ✓ | ✗ | ✗ (use TRT-LLM) | FP16/INT8/FP8 |
| vLLM | ✓ | ✗ | ✓ (PagedAttention) | FP16/INT4 |
| llama.cpp | ✓ | ✓ | ✓ | GGUF (2–8 bit) |
| TensorRT-LLM | ✓ | ✗ | ✓ | FP16/INT8/FP8/INT4 |
| CTranslate2 | ✓ | ✓ | ✓ | INT8/FP16 |

### Batching Strategies

```
Static batching:    Fixed batch size, wait for fill → high latency
Dynamic batching:   Adaptive batch size, queue-based → medium latency
Continuous batch:   Sequences join/leave at each step → max throughput
```

### KV-Cache Optimisations

| Technique | Memory saving | Notes |
|-----------|--------------|-------|
| MQA (Multi-Query Attention) | 50%+ | Architectural change |
| GQA (Grouped-Query Attention) | 30–50% | Architectural change |
| KV-cache INT8 quantization | 50% | No model change |
| PagedAttention | Near 0% waste | vLLM |
| Prefix caching | Variable | Shared prefix across requests |
| H2O / StreamingLLM | Variable | Evict old tokens |

### Latency vs Throughput Trade-offs

```
Goal: Low latency (< 100ms)
  → Small batch size, less GPU utilisation

Goal: High throughput (max requests/sec)
  → Large batch size, continuous batching, quantized model

Goal: Low cost
  → CPU inference with INT8, or quantized model on cheaper GPU (L4 vs A100)
```

---

## 5. Key CLI Commands

```bash
# Launch DDP training
torchrun --nproc_per_node=8 train.py

# Check NCCL version and connectivity
nccl-tests

# Launch with torch elastic (fault tolerant)
torchrun --nproc_per_node=8 --max_restarts=3 train.py

# Convert to ONNX
python -c "import torch; m=torch.load('model.pt'); torch.onnx.export(m, ...)"

# Quantize ONNX model
python -m onnxruntime.quantization --model model.onnx --output model_int8.onnx

# Build TensorRT engine from ONNX
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16

# Launch vLLM server
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B

# Push model to HuggingFace Hub
huggingface-cli upload my-org/my-model ./checkpoint

# Track with DVC
dvc add checkpoints/model.pt && git add checkpoints/model.pt.dvc && git commit
```
