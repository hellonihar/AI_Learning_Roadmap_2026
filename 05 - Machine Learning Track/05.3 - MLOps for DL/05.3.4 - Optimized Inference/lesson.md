# Lesson 05.3.4: Optimized Inference

## Overview

Deploying a deep learning model to production is fundamentally different from training. Inference demands low latency, high throughput, and efficient resource utilisation — often on constrained hardware. This lesson covers the optimisation techniques and tools that bridge the gap between a trained model and a production inference service.

---

## 1. Inference Optimisation Techniques

### 1.1 Quantization for Inference

Quantization reduces numerical precision of weights and activations, producing smaller, faster models.

| Type | Bit-width | Speedup (GPU) | Speedup (CPU) | Accuracy impact |
|------|-----------|---------------|---------------|-----------------|
| FP16 | 16 | 1.5–2× | 1× | Negligible |
| INT8 | 8 | 2–3× | 2–4× | < 1% |
| INT4 | 4 | 3–4× | 3–5× | 1–3% |

**FP16 inference:** Nearly free on Volta+ GPUs (Tensor Cores). Most models can run in FP16 without any calibration.

**INT8 inference:** Requires calibration or QAT. `torch.ao.quantization` provides:
- **Dynamic quantization:** Weights quantized ahead of time; activations quantized on-the-fly per batch. Best for transformer models (linear layers dominate).
- **Static quantization:** Both weights and activations pre-calibrated. Requires representative data. Faster than dynamic.

**INT4 inference:** Emerging standard for LLMs. Methods like GPTQ, AWQ, and GGUF (llama.cpp) allow running 7B–70B models on consumer GPUs or CPU.

### 1.2 Pruning

Remove redundant weights or structures:
- **Unstructured pruning:** Zero out individual weights. Requires sparse matrix support for speedup.
- **Structured pruning:** Remove entire attention heads, layers, or channels. Yields actual speedup on any hardware.
- **2:4 structured sparsity:** NVIDIA Ampere+ GPUs have dedicated hardware for 2:4 sparse patterns — 2× speedup for matrix multiply.

### 1.3 Knowledge Distillation

Replace the large teacher model with a smaller student model trained to mimic its outputs. At inference time, only the student runs.

- **DistilBERT:** 40% smaller, 60% faster, 97% of BERT performance.
- **TinyBERT:** 7.5× smaller, 9.4× faster.
- Distillation can be combined with quantization for cumulative gains.

---

## 2. Inference Engines & Runtimes

### 2.1 ONNX Runtime

ONNX (Open Neural Network Exchange) is an open format for representing models. ONNX Runtime (ORT) is a cross-platform inference engine.

**Benefits:**
- **Graph optimisations:** Node fusion, constant folding, layout optimisation.
- **Execution providers:** Switches between CPU, CUDA, TensorRT, DirectML, OpenVINO, CoreML without code changes.
- **Quantization tooling:** Built-in INT8/FP16 quantization with calibration.

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
outputs = session.run(["output"], {"input": input_tensor})
```

**Conversion path:**
```
PyTorch → torch.onnx.export() → ONNX → ORT (with TensorRT EP)
```

### 2.2 TensorRT

NVIDIA TensorRT optimises models for NVIDIA GPUs. It performs:
- **Layer fusion:** Combines kernels (e.g., Conv+BN+ReLU → one kernel).
- **Precision calibration:** INT8/FP16 with minimal accuracy loss.
- **Kernel auto-tuning:** Selects fastest CUDA kernel for each layer.
- **Memory optimisation:** Pooling and recycling of GPU memory.

```python
import tensorrt as trt

builder = trt.Builder(trt.Logger())
network = builder.create_network()
# ... populate network from ONNX ...
engine = builder.build_engine(network, config)
```

TensorRT typically delivers 2–5× latency improvement over raw PyTorch eager mode.

### 2.3 TorchScript

PyTorch's built-in JIT compilation. Two modes:
- **Tracing:** Runs a dummy input through the model, records operations. Limited if control flow depends on data.
- **Scripting:** Compiles the model via `torch.jit.script`, handles full Python control flow.

```python
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model.pt")
# Load for inference
model = torch.jit.load("model.pt")
```

TorchScript is less aggressive than TensorRT/ORT but requires no extra dependencies.

### 2.4 vLLM & LLM-Specific Optimisations

vLLM is a high-throughput inference engine for LLMs, introducing **PagedAttention**.

**PagedAttention:**
Traditional KV-cache allocation wastes memory (pre-allocates max sequence length per request). PagedAttention manages KV-cache in fixed-size blocks (pages), similar to virtual memory. This eliminates internal fragmentation and enables sharing across requests (e.g., parallel sampling from the same prefix).

**Performance:**
- Up to 24× higher throughput compared to HuggingFace Transformers.
- Near-zero waste in KV-cache memory.
- Supports continuous batching (see below).

**Other LLM inference tools:**
- **TensorRT-LLM:** NVIDIA's LLM-optimised TensorRT variant. Supports in-flight batching, FP8, and multi-GPU tensor parallelism.
- **llama.cpp:** CPU-first inference with GGUF quantized models. Runs on laptops and edge devices.
- **CTranslate2:** Transformer-optimised inference with INT8/FP16 support.

---

## 3. Batching Strategies

### 3.1 Static Batching

Collect requests until a fixed batch size is reached, then run inference. Simple but introduces tail latency (waiting for batch to fill).

### 3.2 Dynamic Batching

The server adjusts batch size based on incoming request rate. Common approach:
- **Queue-based:** Incoming requests join a queue. A scheduler drains the queue every `max_batch_time` or when `max_batch_size` is reached.

### 3.3 Continuous Batching

Transformers process tokens autoregressively. With continuous batching, sequences at different positions in the generation process are batched together. When one sequence finishes, a new one is inserted into the slot.

- **vLLM** implements continuous batching natively.
- **NVIDIA Triton Inference Server** supports it via the "decoupled" backend.
- **HuggingFace TGI** also uses continuous batching.

Continuous batching improves GPU utilisation from ~30% (static) to 80%+ for LLM workloads.

---

## 4. KV-Cache Optimisation

In autoregressive generation (LLMs), the key-value tensors from previous tokens are cached to avoid recomputation:

```
output_t = attention(Q_t, K_{1:t}, V_{1:t})  # K,V cached from previous steps
```

**The KV-cache grows linearly with sequence length** — for a 70B model at 4K context, the cache exceeds 40 GiB per request.

**Optimisation techniques:**

| Technique | Description | Saving |
|-----------|-------------|--------|
| **Multi-Query Attention (MQA)** | Single KV head shared across all query heads. | 50%+ |
| **Grouped-Query Attention (GQA)** | Intermediate between MHA and MQA. | 30–50% |
| **KV-cache quantization** | Store KV vectors in INT8/FP8. | 50% |
| **KV-cache pruning** | Evict less important tokens (e.g., H2O, StreamingLLM). | Variable |
| **PagedAttention** | Block-level memory management. | Near 0% waste |
| **Prefix caching** | Cache KV for shared prefix across requests. | Proportional to prefix hit rate |

---

## 5. Hardware Considerations

### 5.1 GPU Inference

| GPU | VRAM | INT8 TOPS | Best for |
|-----|------|-----------|----------|
| RTX 4090 | 24 GB | 1,320 | Local LLMs (7B–13B quantized) |
| A100 80G | 80 GB | 1,248 | Production LLMs (70B–175B) |
| H100 | 80 GB | 3,958 | Large-scale deployment |
| L4 | 24 GB | 242 | Cost-efficient cloud inference |

**Key metric:** TFLOPS utilisation (not just peak TFLOPS). Continuous batching and good batching strategies improve utilisation from ~20% to >70%.

### 5.2 CPU Inference

CPU inference is slower but cheaper and more accessible. Important for:
- **Edge devices:** No GPU available.
- **Serverless:** Cold starts are faster (no GPU allocation).
- **Cost-sensitive workloads:** CPU cost per inference can be lower.

**Optimisations for CPU:**
- INT8 quantization (especially with VNNI instructions on Ice Lake+).
- `torch.set_num_threads()` and OpenMP tuning.
- ONNX Runtime with `CPUExecutionProvider` and MLAS backend.

### 5.3 Edge Deployment

Deploying models on phones, embedded systems, or IoT devices requires extreme optimisation:

- **TensorFlow Lite / PyTorch Mobile:** Quantized, pruned, and operator-reduced models.
- **ONNX Runtime Mobile:** ARM-compatible build with reduced binary size.
- **Core ML (Apple):** Apple Silicon optimised.
- **Qualcomm SNPE / QNN:** Snapdragon NPU acceleration.
- **ExecuTorch:** PyTorch's new edge runtime (experimental as of 2025).

**Typical edge pipeline:**
```
Full model → Pruning → Distillation → INT8 Quantization → Edge runtime
```

---

## Summary

- **Quantization** (INT8/INT4) is the single highest-impact optimisation for inference.
- **Inference engines** (ONNX Runtime, TensorRT, vLLM) provide 2–10× speedups over eager execution.
- **Continuous batching** maximises GPU utilisation for LLM serving.
- **KV-cache optimisation** is critical for long-context LLMs.
- **Hardware choice** depends on latency, throughput, cost, and deployment environment (cloud vs edge).
