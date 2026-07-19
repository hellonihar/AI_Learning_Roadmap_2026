# 07.3.3 — Quantization

## Overview

Quantization reduces the numerical precision of model weights and activations, shrinking memory footprint and enabling faster inference on commodity hardware. A typical LLM uses FP16 (2 bytes per parameter); INT4 quantization brings this to 0.5 bytes — a 4x memory reduction.

## Fundamentals

Quantization maps a range of floating-point values $[r_{\min}, r_{\max}]$ to a discrete set of integer levels. For symmetric quantisation:

$$
x_q = \text{round}\left( \frac{x}{\Delta} \right), \quad \Delta = \frac{2^{\,b-1} - 1}{\max(|x|)}
$$

where $b$ is the bit width and $\Delta$ is the scale factor. Dequantisation recovers: $\hat{x} = x_q \times \Delta$.

**Post-Training Quantization (PTQ)**: quantise a pre-trained model without retraining. **Quantization-Aware Training (QAT)**: simulate quantisation during training for higher accuracy.

## GPTQ — Generative Pre-Trained Transformer Quantization

**Paper**: *GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers*

GPTQ performs one-shot weight quantisation using an approximation of the Optimal Brain Quantization (OBQ) method. It processes weights column-by-column, quantising each column while updating the remaining unquantised weights to compensate for the error:

$$
w_q = \arg\min_{w_q} \| W_x - w_q x \|^2
$$

GPTQ achieves INT4 quantisation with minimal perplexity loss (often < 0.5 ppl) on billion-parameter models.

## AWQ — Activation-Aware Weight Quantization

**Paper**: *AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration*

AWQ observes that a small fraction of weights (typically 1%) are significantly more important — those corresponding to large activation magnitudes. Rather than quantising all weights uniformly, AWQ applies **per-channel scaling** to protect salient weights:

$$
\hat{W} = \text{quant}(W \cdot s) \cdot s^{-1}
$$

where $s$ is a per-channel scale learned to minimise the quantisation error on a small calibration set. AWQ outperforms GPTQ at the same bit-width, especially at INT4 group size 128.

## GGML / GGUF

GGML and its successor GGUF are file formats and runtime libraries optimised for CPU inference (with GPU offloading). They support a wide range of quantisation levels (q2_k, q3_k, q4_0, q4_k, q5_0, q5_1, q6_k, q8_0) that represent different trade-offs between quality and size.

The GGUF format is used by llama.cpp, the most popular on-device LLM inference engine. For example, Llama 3 8B fits in ~5GB with q4_k_m, running on a consumer laptop.

## bitsandbytes

bitsandbytes provides easy-to-use quantisation for PyTorch models via `bnb.nn.Linear4bit` and `bnb.nn.Linear8bitLt`. The `load_in_4bit` flag in Hugging Face Transformers enables quantised inference with two lines of code:

```python
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b", load_in_4bit=True)
```

bitsandbytes uses **block-wise quantisation** (groups of 64 or 128 weights) and **double quantisation** (quantising the scale factors themselves) to minimise memory overhead.

## Impact on Quality

| Precision | Bits/Param | Memory (7B) | Perplexity (WikiText) |
|-----------|-----------|-------------|----------------------|
| FP16 | 16 | 14 GB | 5.68 |
| INT8 | 8 | 7 GB | 5.69 |
| INT4 | 4 | 3.5 GB | 5.72 |
| INT3 | 3 | 2.6 GB | 6.08 |

Quality degradation at INT4 is minimal for most tasks, making it the default for deployment.

## Practical Context

Quantization is the primary technique for deploying LLMs on consumer GPUs, edge devices, and CPUs. The choice between GPTQ, AWQ, and GGUF depends on your hardware and deployment target — AWQ for GPU inference, GGUF for CPU, GPTQ for maximum throughput.
