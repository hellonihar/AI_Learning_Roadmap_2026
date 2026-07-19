# Lesson 05.3.2: Checkpointing

## Overview

Training large deep learning models can take days or weeks on hundreds of GPUs. Hardware failures, preemption on spot instances, and job time limits are inevitable. Checkpointing — periodically saving the full training state — is the primary mechanism for fault tolerance and training reproducibility.

This lesson covers what to save, when to save, how to manage checkpoints in distributed settings, and how to recover from failures.

---

## 1. What Is in a Checkpoint?

A training checkpoint must capture everything needed to reconstruct the training loop exactly. The minimal set includes:

| Component | Why it matters |
|-----------|----------------|
| **Model weights** (state_dict) | The learned parameters; without them there is no model. |
| **Optimizer state** | Momentum buffers (SGD), first/second moments (Adam/AdamW). Without these, training would restart with default momentum — wasted steps. |
| **Learning rate scheduler** | Current LR, epoch/step counters. Without it, the LR schedule restarts, potentially destabilising training. |
| **Epoch / global step** | Position in training. Needed for dataset resumption (shuffle order), logging, and evaluation scheduling. |
| **RNG state** | `torch.random`, NumPy, Python random. Ensures data augmentation and dropout patterns are reproducible. |

**Optional but recommended:**
- DataLoader iterator state (for exact sample-level recovery).
- Gradient scaler state (if using AMP — `torch.cuda.amp.GradScaler`).
- Model EMA weights (exponential moving average used at inference time).

---

## 2. Checkpointing Frequency

### 2.1 Periodic (Time-Based) Checkpoints

Save every N epochs or every M training steps.

- **Pros:** Simple to implement; bounded recovery cost (at most N steps of work lost).
- **Cons:** May save many intermediate states (storage cost).
- **Best practice:** Save every 30–60 minutes of training time. For a 100-hour job, 100–200 checkpoints.

**Retention policy:** Keep the last K checkpoints + every Nth checkpoint for long-range history. Automatically delete older ones unless they are needed for analysis.

### 2.2 Best-Model Checkpoints

Monitor a validation metric (e.g., validation loss, accuracy, BLEU) and save whenever the metric improves.

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    save_checkpoint(model, optimizer, scheduler, epoch, "best_model.pt")
```

**Important:** Best-model checkpoints should save only the model weights, not the full optimizer state, to minimise storage. Maintain a separate periodic checkpoint for resume purposes.

### 2.3 Last / Final Checkpoint

Save when training finishes (either normally or via a signal handler). The final checkpoint is the deliverable for downstream inference or fine-tuning.

---

## 3. Distributed Checkpointing

In distributed training, checkpoints must account for the fact that the model may be sharded across ranks.

### 3.1 DDP (Data Parallelism)

In DDP, every rank has an identical copy of the model. The simplest strategy:
1. **Rank 0** saves the full model (rank 0's copy is identical to all others because gradients are synchronised).
2. All other ranks skip saving.

**Problem:** For very large models, rank 0 may OOM when materialising the full state dict. Solution: `torch.save` on CPU after moving state dict to CPU.

### 3.2 FSDP / ZeRO-3 (Sharded Parameters)

In FSDP or DeepSpeed ZeRO-3, parameters are sharded — no single rank holds the full model.

**Two approaches:**

**A. Full state dict consolidation (rank 0 gathers everything):**
```python
state = model.state_dict()  # gathers shards on CPU
if rank == 0:
    torch.save(state, "checkpoint.pt")
```
Memory overhead: rank 0 must have enough CPU RAM for the full model.

**B. Distributed checkpointing (each rank saves its shard):**
```python
# Each rank saves only its own shard
torch.save({
    "model": model.local_state_dict(),   # shard only
    "optimizer": optimizer.state_dict(), # shard only
    "epoch": epoch,
}, f"checkpoint_rank{rank}.pt")
```
On resume, each rank loads its own shard. This has no memory overhead and scales to any model size.

**DeepSpeed provides `deepspeed.checkpointing`** and PyTorch provides `torch.distributed.checkpoint` for this purpose.

### 3.3 Avoiding Metadata Skew

- Save the world size (`dist.get_world_size()`) in the checkpoint.
- On resume, verify that the world size matches. If world size changed, you must handle tensor re-sharding (map old shards to new ranks).

---

## 4. Failure Recovery

### 4.1 Hardware Failures

In long-running jobs, GPU or node failures are statistically likely.

Common recovery strategies:

| Strategy | Description | Overhead |
|----------|-------------|----------|
| **Manual restart** | Operator detects failure, restarts job from latest checkpoint. | High (human in loop) |
| **TorchElastic / fault-tolerant launch** | `torchrun` with `--max_restarts=N`. On rank failure, surviving ranks rendezvous and restart. | Low (auto) |
| **DeepSpeed elastic** | Automatically saves and loads sharded checkpoints on membership change. | Low (auto) |

**TorchElastic example:**
```bash
torchrun --nproc_per_node=8 --max_restarts=3 train.py
```
If a rank dies, the remaining ranks wait for it to come back (in Kubernetes the pod is rescheduled). Training resumes from the last saved checkpoint.

### 4.2 Preemption (Spot Instances)

Cloud spot instances can be terminated at any time (2-minute warning). Best practices:
- Save checkpoints every 10–15 minutes (aggressive).
- Listen for SIGTERM and force a checkpoint save in the signal handler.
- Resume automatically when new spot capacity is available.

---

## 5. Storage Management

### 5.1 Local vs Shared Filesystem

| Storage type | Appropriate for | Concerns |
|-------------|----------------|----------|
| Local SSD (NVMe) | Fast writes; good for single-node training. | Lost on node failure; must sync to remote. |
| NFS / shared FS | All ranks can read/write. | Contention at scale; metadata bottleneck. |
| Object store (S3, Blob) | Durable; pay-per-use. | Higher latency; no POSIX locking. |

**Recommended architecture:** Save to local SSD first, then async copy to object store as a sidecar process.

### 5.2 Deduplication and Compression

- Use `torch.save` with `pickle_protocol=5` for faster serialisation.
- Apply `zip` compression (`torch.save(..., _use_new_zipfile_serialization=True)` in older versions; new default).
- Deduplicate shared tensors (e.g., tied embeddings) to avoid double counting.

---

## 6. Implementation Skeleton

```python
def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "rng_state": torch.get_rng_state(),
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]
```

For FSDP, use `state_dict_type` context manager:
```python
with fsdp_model.state_dict_type(StateDictType.FULL_STATE_DICT):
    state = fsdp_model.state_dict()
```

---

## Summary

- **Checkpoints contain** model weights, optimizer state, scheduler, epoch counter, and RNG state.
- **Save periodically** (every 30–60 min) + best-model saves on validation improvement.
- **Distributed saving** requires handling sharded parameters (FSDP/ZeRO-3) via per-rank shard saving or CPU consolidation.
- **Failure recovery** is automated via TorchElastic or DeepSpeed's fault-tolerant launcher.
- **Storage strategy** should prioritise speed (local SSD) with durability (async copy to blob storage).

Next lesson: **05.3.3 — Large Model Storage**, covering weight formats, quantization, compression, and versioning.
