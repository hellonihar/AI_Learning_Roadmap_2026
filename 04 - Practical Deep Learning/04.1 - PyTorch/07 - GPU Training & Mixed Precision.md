# GPU Training & Mixed Precision

GPUs accelerate deep learning by orders of magnitude. PyTorch makes GPU training straightforward, with tooling for both single-GPU and multi-GPU setups.

## Device Management

The standard pattern puts both model and data on the same device:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)

for inputs, targets in dataloader:
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
```

**Never mix devices:** operations between CPU and GPU tensors raise errors. Always move data to the model's device before forward.

### Checking GPU Availability

```python
torch.cuda.is_available()       # bool
torch.cuda.device_count()       # number of GPUs
torch.cuda.current_device()     # index of current GPU
torch.cuda.get_device_name(0)   # e.g., "NVIDIA A100 80GB"

# Set device globally
torch.cuda.set_device(0)
```

### CUDA Semantics

- PyTorch allocates GPU memory lazily and caches it for reuse.
- Use `torch.cuda.empty_cache()` to free unused memory (rarely needed).
- Monitor memory with `nvidia-smi` or `torch.cuda.memory_summary()`.

## DataParallel (DP)

Simple multi-GPU wrapper — splits the batch across GPUs:

```python
model = nn.DataParallel(model)       # replicates model on all GPUs
outputs = model(inputs)              # automatically splits batch
```

**Limitations:** DP runs all GPUs on one process with one Python thread. The primary GPU does extra work (scattering, gathering). For most cases, **DDP is preferred.**

## DistributedDataParallel (DDP) — Conceptual

DDP spawns one process per GPU with minimal communication overhead:

```python
# On each process (typically launched with torchrun)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Dataloader with distributed sampler
sampler = DistributedSampler(dataset)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

| Feature | DataParallel | DistributedDataParallel |
|---------|-------------|------------------------|
| Process model | Single process, multi-threaded | Multi-process, one per GPU |
| Speed | Slower (GIL contention) | Faster (independent processes) |
| Scalability | 2-4 GPUs | 8+ GPUs, multi-node |
| Code change | One-line wrapper | Requires launch script |

**Rule of thumb:** Use DDP for any multi-GPU training. DP is a legacy convenience for quick prototyping.

## Mixed Precision (AMP)

Mixed precision trains with `float16` (half) where safe and `float32` where needed, using `torch.cuda.amp`. Benefits: **~2× memory savings** and **up to 3× speedup** on modern GPUs (Tensor Cores).

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()                       # prevents underflow

for inputs, targets in dataloader:
    inputs, targets = inputs.to(device), targets.to(device)

    optimizer.zero_grad()

    with autocast():                        # automatic mixed precision
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()           # scale loss to prevent underflow
    scaler.step(optimizer)                  # unscale gradients, then step
    scaler.update()                         # adjust scale factor
```

### How Mixed Precision Works

| Operation | Precision | Rationale |
|-----------|-----------|-----------|
| Forward pass | float16 | Matrix multiplications use Tensor Cores (8× speedup on A100) |
| Loss computation | float16/32 | Depends on operation |
| Backward pass | float16 | Gradient computation through matmul |
| Gradient update | float32 | Master weights kept in float32 for stability |
| Loss scaling | float32 | Scales small gradients into fp16 range |

### GradScaler

The `GradScaler` prevents **gradient underflow** — gradients smaller than ~6e-8 (fp16 minimum) become zero. It:

1. **Scales** the loss up before `backward()`, so gradients are larger.
2. **Unscales** before `optimizer.step()`.
3. **Adjusts** the scale factor: increases if no overflow, decreases on overflow.

### When to Use AMP

- **Always** on Volta, Turing, Ampere, Hopper, or newer NVIDIA GPUs.
- **Always** for large models (ViT, GPT, BERT).
- **Skip** if training is CPU-bound or data pipeline is the bottleneck.
- **Skip** on older GPUs (Kepler, Maxwell) — no Tensor Cores.

### Full Training Loop with AMP

```python
model.train()
scaler = GradScaler()

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Validation — no AMP needed
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            ...
```

## Summary

```python
# Single GPU
model = MyModel().to(device)
# Multi-GPU (DDP)
model = DDP(MyModel().to(local_rank), device_ids=[local_rank])
# Mixed precision
with autocast(): loss = criterion(model(x), y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

> **Bottom line:** Use `model.to(device)` for single-GPU. Use DDP for multi-GPU (skip DP). Enable AMP for 2× memory savings and faster training on modern GPUs. These three techniques together let you train models that would otherwise be impossible on your hardware.
