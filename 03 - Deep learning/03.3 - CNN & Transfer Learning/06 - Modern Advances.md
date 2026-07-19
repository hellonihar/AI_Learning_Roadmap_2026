# Modern Advances in Computer Vision

## Depthwise Separable Convolutions (MobileNet)

Standard convolution simultaneously maps spatial correlations and cross-channel correlations. Depthwise separable convolution splits this into two operations, dramatically reducing computation.

### Standard 3×3 Conv
```
Input:  D_F × D_F × M
Kernel: D_K × D_K × M × N
Cost:   D_K · D_K · M · N · D_F · D_F
```

### Depthwise Separable (MobileNet)

**Step 1 — Depthwise Conv**: A single filter *per input channel*.

```
Depthwise: D_K × D_K × 1 × M  →  Cost: D_K · D_K · M · D_F · D_F
```

**Step 2 — Pointwise Conv (1×1)**: Combine outputs across channels.

```
Pointwise: 1 × 1 × M × N  →  Cost: M · N · D_F · D_F
```

### Cost Comparison

```
Standard:   D_K · D_K · M · N · D_F · D_F
Separable:  D_K · D_K · M · D_F · D_F  +  M · N · D_F · D_F
Reduction:  1/N + 1/D_K²
```

For 3×3 kernels (D_K = 3): reduction factor ≈ 8–9×.

### MobileNet Architecture

MobileNetV1 uses depthwise separable convolutions throughout, with width multiplier α and resolution multiplier ρ for further scaling. MobileNetV3 combines NAS (Neural Architecture Search) with squeeze-and-excitation modules for state-of-the-art efficiency.

## EfficientNet (Compound Scaling)

### Key Insight
Previous work scaled CNNs along a single dimension:
- **Depth** (more layers — ResNet)
- **Width** (more channels — WideResNet)
- **Resolution** (larger input — higher-resolution ImageNet)

EfficientNet found that these dimensions are **not independent**. Compound scaling balances all three:

```
ϕ = compound coefficient
depth:   d = α^ϕ
width:   w = β^ϕ
res:     r = γ^ϕ
subject to α · β² · γ² ≈ 2  (computation cost constraint)
```

### Results

| Model | Parameters | FLOPS | ImageNet Top-1 |
|---|---|---|---|
| EfficientNet-B0 | 5.3M | 0.4B | 77.1% |
| EfficientNet-B7 | 66M | 37B | 84.3% |

EfficientNet-B7 achieves similar accuracy to GPipe with 1/8th the parameters and matches ResNet-50's cost with 3× fewer parameters.

## Vision Transformers (ViT)

### Concept
Replace convolutions entirely with a Transformer architecture, treating image patches as tokens (like words in NLP).

### ViT Pipeline

```
1. Split image into patches (e.g., 16×16 each → 196 patches for 224×224 image).
2. Flatten each patch to a vector, project with a linear layer.
3. Add positional embeddings.
4. Add a [CLS] token (as in BERT) to aggregate global information.
5. Feed through standard Transformer encoder layers (MSA + MLP, LayerNorm, residual).
6. Use [CLS] output for classification.
```

### Key Properties
- **No convolutions** — pure self-attention.
- **Data hungry**: ViT needs large-scale pre-training (ImageNet-21k or JFT-300M) to match CNN performance. On small data it underperforms.
- **Inductive bias**: CNNs have strong locality bias (convolution, pooling). ViT has weaker inductive bias — relies on learning from data.

### Hybrid Models
Modern approaches (ConvNeXt, MaxViT, CoAtNet) combine convolution and attention, using conv layers for early stages (high resolution) and attention for later stages (low resolution, high semantics).

### Modern Adoption
While ViT and its variants (DeiT, Swin) now match or exceed CNNs on many benchmarks, convolution remains dominant for:
- Mobile / edge deployment (MobileNet, EfficientNet-Lite)
- Small datasets (transfer learning with ResNets)
- Pixel-level tasks (segmentation, detection) where locality is critical
