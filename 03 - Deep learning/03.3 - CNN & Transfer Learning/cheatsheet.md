# Cheatsheet: CNN & Transfer Learning

## Convolution Formula

```
Discrete 2D convolution:
S(i, j) = Σₘ Σₙ I(i + m, j + n) · K(m, n)

Multi-channel (RGB):
S(i, j, k) = Σₘ Σₙ Σ_c I(i+m, j+n, c) · K(m, n, c, k) + b(k)
```

## Output Size Calculation

```
W_out = ⌊(W_in - K + 2P) / S⌋ + 1
H_out = ⌊(H_in - K + 2P) / S⌋ + 1
D_out = number of filters
```

| Condition | Padding P | Output Size (S=1) |
|---|---|---|
| Valid | 0 | W_in - K + 1 |
| Same (K odd) | (K - 1) / 2 | W_in |
| Same (K even) | (K - 1) / 2 | W_in |

## Receptive Field

```
RF_(l) = RF_(l-1) + (K_l - 1) · ∏_{i=1}^{l-1} S_i
```

Two 3×3 conv layers (stride 1) → RF = 5.  
Three 3×3 conv layers (stride 1) → RF = 7.

## Parameter Count

```
Conv layer params = K × K × C_in × C_out + C_out (bias)
FC layer params  = N_in × N_out + N_out (bias)
```

## Pooling Types

| Type | Operation | Output per Window | Common Size |
|---|---|---|---|
| Max | max(patch) | 1 value | 2×2, stride 2 |
| Average | mean(patch) | 1 value | 2×2, stride 2 |
| Global Avg | mean(H, W) per channel | 1 value per channel | Full spatial |

## Activation Functions

```
ReLU:    f(x) = max(0, x)
Leaky:   f(x) = max(αx, x)        α ≈ 0.01
Softmax: f(x_i) = e^x_i / Σ_j e^x_j
Sigmoid: f(x) = 1 / (1 + e^(-x))
```

## Architecture Comparison

| Model | Year | Layers | Params | Key Feature | Top-5 Error |
|---|---|---|---|---|---|
| LeNet-5 | 1998 | 7 | 60K | Conv+Pool+FC template | ~0.9% (MNIST) |
| AlexNet | 2012 | 8 | 60M | ReLU, Dropout, GPU | 15.3% |
| VGG-16 | 2014 | 16 | 138M | 3×3 conv stacks | 7.3% |
| GoogLeNet | 2014 | 22 | 5M | Inception module | 6.7% |
| ResNet-50 | 2015 | 50 | 25.6M | Skip connections | 3.57% |
| DenseNet-121 | 2017 | 121 | 8M | Dense connectivity | ~3.7% |

## Transfer Learning Strategy

| Dataset | Similar to Source | Different from Source |
|---|---|---|
| Small (< 1K/class) | Feature extraction | Feature extraction + reg |
| Medium (1K–10K/class) | Fine-tune top 1/2 | Fine-tune all (low LR) |
| Large (> 10K/class) | Fine-tune all | Fine-tune all |

### Process

```
1. Load pre-trained model (no head)
2. Add new classifier head
3. Freeze base, train head (warm-up)
4. Unfreeze base, train with low LR (1/10th of default)
```

## Data Augmentation

```
Geometric:  RandomFlip, RandomRotation, RandomCrop, RandomTranslate
Color:      ColorJitter(brightness, contrast, saturation, hue)
Advanced:   CutOut, MixUp, CutMix, RandAugment
```

## Depthwise Separable Conv Cost

```
Standard:  D_K² · M · N · D_F²
Separable: D_K² · M · D_F²  +  M · N · D_F²
Ratio:     1/N + 1/D_K²
```

For 3×3: ~8-9× fewer operations.

## Common Kernel Sizes

- 1×1: Dimensionality reduction, channel mixing
- 3×3: Standard — used in VGG, ResNet, EfficientNet
- 5×5: Larger RF — used in Inception
- 7×7: Initial layer — used in ResNet, AlexNet

## Normalization Layers

```
BatchNorm:  μ, σ per batch during training, running stats at inference
LayerNorm:  μ, σ per sample across features (used in Transformers)
```

## Gradient Flow

```
∂L/∂K = I * ∂L/∂S       (convolution of input with output gradient)
∂L/∂I = K_rot180 * ∂L/∂S  (full convolution with 180° rotated kernel)
```

## Transfer Learning: Key Terms

- **Pre-training**: Training on large dataset (e.g., ImageNet) — learns generic features.
- **Fine-tuning**: Continued training on target data with low LR — adapts to target domain.
- **Feature extraction**: Frozen pre-trained weights act as fixed encoder.
- **Domain adaptation**: Adjusting model when source and target distributions differ.
- **Catastrophic forgetting**: Loss of pre-trained knowledge when fine-tuning too aggressively.
