# Exercises: MLOps for Deep Learning

## Exercise 1: Selecting a Distributed Strategy

**Scenario:** You need to train a 7B-parameter transformer model on 32 NVIDIA A100 GPUs (8 nodes × 4 GPUs). Per-node NVLink bandwidth is 600 GB/s; inter-node InfiniBand is 200 Gb/s (25 GB/s). The model barely fits on a single A100 in FP16.

**Questions:**
1. Which parallelism strategy (or combination) would you recommend?
2. Why is pure DDP not viable here?
3. Should you use tensor parallelism? Why or why not?

**Answer:**
1. A combination of DeepSpeed ZeRO-3 (data + sharded parameters) with optional pipeline parallelism. ZeRO-3 alone shards the model parameters across all 32 GPUs, so each GPU holds only ~220 MB of parameters. This gives plenty of room for activations and optimizer states.
2. Pure DDP requires the full model on every GPU. At 7B × 2 bytes (FP16) = 14 GB just for parameters, plus optimizer states (28 GB for Adam) and activations, an A100's 80 GB would be constrained — especially with large batch sizes or long sequences.
3. Tensor parallelism is not strictly necessary because the model fits on one GPU. Tensor parallelism would add all-reduce overhead on every layer, and its benefits (reducing per-GPU memory further) are not needed here. It would also require careful intra-node placement (NVLink). Save TP for models > 10B params.

---

## Exercise 2: ZeRO Stage Comparison

**Scenario:** You are training a 13B-parameter model on 64 GPUs. Adam optimizer uses 4 bytes per parameter for its states (2 for moment, 2 for variance). Gradient storage is 2 bytes per parameter.

**Calculate the memory per GPU for:**
1. DDP (no ZeRO)
2. ZeRO-1
3. ZeRO-2
4. ZeRO-3

Assume FP16 parameters (2 bytes each), and no activation memory.

**Answer:**

Per-parameter memory costs:
- Parameters: 2 bytes
- Gradients: 2 bytes
- Adam states: 4 bytes
- **Total per parameter: 8 bytes**

Total model: 13B × 8 bytes = 104 GB

| Strategy | Parameters | Gradients | Optimizer | Per GPU (104 GB / divisor) |
|----------|-----------|-----------|-----------|---------------------------|
| **DDP** | Full (26 GB) | Full (26 GB) | Full (52 GB) | **104 GB** — impossible |
| **ZeRO-1** | Full (26 GB) | Full (26 GB) | Sharded (52/64 = 0.81 GB) | **~52.8 GB** |
| **ZeRO-2** | Full (26 GB) | Sharded (26/64 = 0.41 GB) | Sharded (0.81 GB) | **~27.2 GB** |
| **ZeRO-3** | Sharded (26/64 = 0.41 GB) | Sharded (0.41 GB) | Sharded (0.81 GB) | **~1.6 GB** |

ZeRO-3 reduces per-GPU memory by ~65× compared to DDP, at the cost of increased communication.

---

## Exercise 3: Checkpoint Resume Bug

**Code:**

```python
def train():
    model = MyModel()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    start_epoch = 0

    if os.path.exists("checkpoint.pt"):
        ckpt = torch.load("checkpoint.pt")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]

    for epoch in range(start_epoch, 100):
        train_one_epoch(model, optimizer)
        scheduler.step()
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }, "checkpoint.pt")
```

**Find the bugs (3 issues).**

**Answer:**
1. **Scheduler state not saved/loaded.** When resuming, the scheduler starts from its initial state (LR = 1e-4) instead of the correct LR for `start_epoch`. The scheduler must be saved and loaded alongside model and optimizer.
2. **Epoch index off-by-one.** The saved `epoch` is the one just completed, but the code starts at `start_epoch` (the saved value). This re-runs the last completed epoch. Fix: save `epoch + 1` as the next epoch to start from, or start with `start_epoch + 1`.
3. **RNG state not saved.** Data augmentation and dropout patterns will differ after resume, breaking reproducibility. Save `torch.get_rng_state()` and restore it on load.

---

## Exercise 4: ONNX Export Failure

**Scenario:** You export a model to ONNX and try to run it with TensorRT, but the engine build fails with:

```
ERROR: ../builder/onnx_parser.cpp:XX: While parsing node
[Node type: "Gather", name: "gather_17"]: Unsupported ONNX data type: INT64
```

**What is the likely cause and how do you fix it?**

**Answer:**
The model uses `torch.argmax`, `torch.sum` with `dtype=torch.int64`, or indexing operations that produce INT64 tensors. TensorRT does not support INT64 by default.

**Fixes:**
1. In the PyTorch model, cast index tensors to INT32 before export:
   ```python
   indices = indices.to(torch.int32)
   ```
2. Or cast before ONNX export: add `torch.onnx.export(..., dynamic_axes=...)` with explicit type casting.
3. If the INT64 nodes are internal (not inputs/outputs), use ONNX Runtime's TensorRT execution provider (it handles casts automatically) instead of native TensorRT.
4. Use `trtexec --onnx=model.onnx --int32` to force INT32 support (if available in your TensorRT version).

---

## Exercise 5: Comparing Inference Engines

**Scenario:** You need to deploy a BERT-large model for a real-time NLP API with the following requirements:
- p99 latency < 50 ms
- Throughput > 1000 requests/second
- Deployed on NVIDIA L4 GPUs (24 GB VRAM)

**Compare these approaches and recommend one:**
1. PyTorch eager mode with FP16
2. ONNX Runtime with INT8 dynamic quantization
3. TensorRT with FP16
4. TensorRT with INT8 (QAT-calibrated)

**Answer:**

| Approach | Est. p99 latency | Throughput | Effort |
|----------|-----------------|-----------|--------|
| PyTorch eager + FP16 | ~80–120 ms | ~500 req/s | Low (no conversion) |
| ONNX Runtime + INT8 | ~30–50 ms | ~1,500 req/s | Medium (conversion + quant) |
| TensorRT + FP16 | ~20–30 ms | ~2,500 req/s | Medium (conversion) |
| TensorRT + INT8 (QAT) | ~10–20 ms | ~5,000 req/s | High (retrain with QAT) |

**Recommendation:** ONNX Runtime with INT8 dynamic quantization. It meets both latency and throughput requirements with moderate effort. TensorRT + FP16 would offer better performance if the team has the engineering bandwidth. TensorRT + INT8 (QAT) is overkill for BERT-large and requires retraining.

---

## Exercise 6: Continuous Batching for LLMs

**Scenario:** You are serving a 70B-parameter LLM on 2× A100 GPUs with tensor parallelism. User requests have varying prompt lengths (100–2000 tokens) and generate 50–500 tokens each. With static batching (batch size = 4), GPU utilisation is 25% and p95 latency is 8 seconds.

**Questions:**
1. Why is utilisation so low with static batching?
2. How does continuous batching improve this?
3. Which inference engine would you recommend?

**Answer:**
1. **Low utilisation causes:**
   - Padding waste: Short sequences padded to max sequence length, wasting compute.
   - Idle time: After all 4 sequences finish generation, the GPU waits for the next batch to fill (queue drain latency).
   - Uneven generation lengths: One long generation holds up the entire batch; shorter sequences finish early but the batch slot is wasted.
   - Batch size limited by peak memory (KV-cache for 4 × max generation length), forcing small batches.

2. **Continuous batching improvement:**
   - Sequences enter/leave the batch at each step (not batch boundaries).
   - When a sequence finishes, a new sequence immediately enters the freed slot.
   - KV-cache is managed per-sequence (PagedAttention), eliminating padding.
   - GPU utilisation can reach 80%+ because there is always work available.
   - Latency improves because requests don't wait for batch formation.

3. **Recommendation:** Use **vLLM** with tensor parallelism across 2 GPUs. For production with enterprise requirements (monitoring, multi-model, model switching), **NVIDIA Triton Inference Server** with TensorRT-LLM backend. Both support continuous batching natively and include PagedAttention for efficient KV-cache management.
