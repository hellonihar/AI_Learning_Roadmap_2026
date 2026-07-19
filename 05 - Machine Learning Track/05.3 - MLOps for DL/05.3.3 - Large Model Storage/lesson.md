# Lesson 05.3.3: Large Model Storage

## Overview

A single LLM checkpoint can exceed 700 GiB (e.g., Llama 3.1 405B in FP16). Storing, versioning, and distributing such large artifacts requires careful planning around numerical formats, compression, quantization, and storage infrastructure.

This lesson covers weight formats, quantization techniques, compression strategies, and tooling for versioning large models.

---

## 1. Model Weight Formats

### 1.1 Float Point Formats

| Format | Bits | Exponent | Mantissa | Range | Precision | Storage (1B params) |
|--------|------|----------|----------|-------|-----------|---------------------|
| **FP32** | 32 | 8 | 23 | ±3.4×10³⁸ | High | ~4 GB |
| **FP16** | 16 | 5 | 10 | ±65,504 | Moderate | ~2 GB |
| **BF16** | 16 | 8 | 7 | ±3.4×10³⁸ | Low (same range as FP32) | ~2 GB |
| **INT8** | 8 | — | — | -128 to 127 | Low | ~1 GB |
| **FP8 (E4M3/E5M2)** | 8 | 4 or 5 | 3 or 2 | ±448 / ±57,344 | Very low | ~1 GB |

**Key insight:** BF16 preserves the full dynamic range of FP32 (important for gradient stability during training) while halving memory. Most large models today are trained in mixed FP16/BF16 precision.

### 1.2 Mixed Precision Training

AMP (Automatic Mixed Precision) stores weights in FP16/BF16 but keeps a **master copy in FP32** for the optimizer (required for AdamW — the FP32 copy absorbs the tiny update steps).

| Component | Precision |
|-----------|-----------|
| Forward/backward | FP16 or BF16 |
| Weights (training) | FP32 master + FP16/BF16 working copy |
| Gradients | FP16 |
| Optimizer states | FP32 |
| Inference weights | FP16, BF16, or INT8 |

### 1.3 SafeTensors vs Pickle

| Format | Security | Speed | Lazy loading |
|--------|----------|-------|-------------|
| **Pickle** (`torch.save`) | Unsafe — arbitrary code execution | Slow for large files | No |
| **SafeTensors** (`safetensors`) | Safe — no code execution | Fast (memory-mapped, zero-copy) | Yes |
| **HDF5** | Safe | Moderate | Yes |

**Recommendation:** Use SafeTensors for distribution. HuggingFace Hub now recommends it as the default.

---

## 2. Quantization

Quantization reduces the bit-width of model weights, enabling faster inference and smaller storage at the cost of accuracy.

### 2.1 Post-Training Quantization (PTQ)

Apply quantization after training is complete. No additional training needed.

**Types:**
- **Weight-only quantization:** Only weights are quantized (e.g., INT4, INT8). Activations remain FP16. Common for LLM inference.
- **Weight + activation quantization:** Both weights and activations quantized. Necessary for integer-only hardware (e.g., Edge TPU, some NPUs).
- **Per-tensor vs per-channel:** Per-channel (axis-wise) quantization preserves more accuracy by scaling each output channel independently.

**Calibration:** A small calibration dataset is run through the model to determine optimal scaling factors (min/max ranges for each tensor).

**Popular PTQ frameworks:**
- `torch.ao.quantization` (PyTorch native)
- `bitsandbytes` (LLM.int8(), NF4)
- `GPTQ` (post-training quantization for LLMs)
- `AWQ` (activation-aware weight quantization)
- `GGML` / `GGUF` (CPU-friendly quantization formats for llama.cpp)

| Method | Bits | Perplexity increase (WikiText-2, Llama 7B) |
|--------|------|------------------------------------------|
| FP16 (baseline) | 16 | — |
| INT8 (RTN) | 8 | ~0.1 |
| INT4 (GPTQ) | 4 | ~0.5 |
| INT4 (AWQ) | 4 | ~0.4 |
| NF4 (QLoRA) | 4 | ~0.5 |
| INT3 (GPTQ) | 3 | ~2.0 |
| INT2 | 2 | ~5.0+ |

RTN = round-to-nearest (naive rounding without calibration).

### 2.2 Quantization-Aware Training (QAT)

Simulate quantization during training so the model learns to compensate for precision loss.

```python
# PyTorch QAT example
model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
model = torch.ao.quantization.prepare_qat(model, inplace=True)
# ... training loop ...
model = torch.ao.quantization.convert(model, inplace=True)
```

QAT typically recovers most of the accuracy lost by PTQ, especially at very low bit-widths (INT4, INT3). The trade-off is the additional training cost.

---

## 3. Model Compression

### 3.1 Pruning

Remove weights or neurons that contribute least to model output.

| Type | Description | Compression |
|------|-------------|-------------|
| **Unstructured pruning** | Zero out individual weights (sparse tensors). | 50–90% sparsity |
| **Structured pruning** | Remove entire channels, heads, or layers. | 20–50% |
| **Magnitude pruning** | Remove weights with smallest absolute values. | Variable |
| **Movement pruning** | Remove weights based on gradient signal during fine-tuning. | Better accuracy at high sparsity |

Structured pruning yields actual speedup on standard hardware (because dense matrix ops become smaller). Unstructured pruning requires sparse hardware support (e.g., NVIDIA Ampere sparse tensor cores, 2:4 sparsity).

### 3.2 Knowledge Distillation

Train a smaller **student** model to mimic a larger **teacher** model.

```
student_logits = student(x)
teacher_logits = teacher(x)  # frozen
loss = KL_div(student_logits, teacher_logits) + CE(student_logits, y_true)
```

Distillation can achieve 90%+ of teacher performance with 10–50% of the parameters. Notable examples: DistilBERT (40% smaller, 97% performance), TinyBERT, MiniLM.

### 3.3 Combined Pipeline

Many production pipelines combine techniques:
```
Full model → Pruning → Distillation → PTQ → Deploy (INT4/INT8)
```

Each compression stage compounds storage savings while managing accuracy loss.

---

## 4. Storage Strategies

### 4.1 Sharding

For models too large to fit in a single file, split checkpoints into shards.

- **HuggingFace sharding:** `save_pretrained(save_directory, max_shard_size="5GB")` creates `model-00001-of-00003.safetensors`, `model-00002-of-00003.safetensors`, etc.
- DeepSpeed checkpointing also produces per-rank shards.

Sharding enables:
- Parallel download from object storage.
- Lazy loading (load only needed shards).
- Streaming (begin inference before full download).

### 4.2 Object Store vs Filesystem

| Factor | Object store (S3, GCS, Azure Blob) | POSIX filesystem |
|--------|-------------------------------------|------------------|
| Durability | 11 nines (automatic replication) | Depends on RAID/backup |
| Throughput | Scales with concurrent connections | Limited by disk/NFS |
| Latency | Higher (HTTP round trip) | Low |
| Metadata ops | Slow (list objects) | Fast (ls/stat) |
| Locking | No native locking | Advisory/file locks |
| Cost | Pay per GB stored + egress | Upfront hardware |

**Best practice:** Use object store as the source of truth; cache frequently accessed files on local SSD.

### 4.3 Content-Addressable Storage

Use content hashes (SHA256) for checkpoint filenames: `model_abc123def...safetensors`. This deduplicates identical checkpoints and ensures integrity verification.

---

## 5. Versioning Large Models

### 5.1 Git LFS (Large File Storage)

Git LFS replaces large files with text pointers in the repo and stores the actual content on a remote server.

```bash
git lfs track "*.safetensors"
git lfs track "*.pt"
```

**Limitations:**
- Per-repo bandwidth quota (e.g., 1 GB/month free on GitHub).
- No semantic diff for binary model files.
- Not designed for model-specific metadata (training config, eval results).

### 5.2 HuggingFace Hub

Dedicated model registry with:
- Versioned model uploads (semantic tags: `v1.0`, `v2.0`).
- Built-in SafeTensors support.
- Model card (README.md) with config, eval results, license.
- Git-based under the hood (Git LFS for weights).
- Community features (discussions, PRs for models).

```bash
huggingface-cli upload my-model ./checkpoint --repo-type model
```

### 5.3 DVC (Data Version Control)

DVC stores metadata in Git and data in a remote store (S3, GCS, etc.).

```bash
dvc add checkpoints/model.pt
git add checkpoints/model.pt.dvc .gitignore
git commit -m "add model checkpoint"
dvc push
```

DVC enables lightweight Git repos while tracking model lineage alongside code.

---

## Summary

- **Weight formats** trade range/precision for memory: BF16 is the training sweet spot; INT8/INT4 dominate inference.
- **Quantization** (PTQ or QAT) compresses models 2–4× with minimal accuracy loss.
- **Pruning and distillation** further reduce model size and compute.
- **Sharding** enables handling of multi-hundred-GB checkpoints.
- **Object storage** with SafeTensors format is the recommended delivery mechanism.
- **HuggingFace Hub** and **DVC** provide versioning and collaboration for large models.

Next lesson: **05.3.4 — Optimized Inference**, covering deployment optimisation techniques and hardware considerations.
