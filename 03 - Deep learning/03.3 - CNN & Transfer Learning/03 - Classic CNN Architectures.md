# Classic CNN Architectures

## LeNet-5 (1998)

**Authors**: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner  
**Paper**: *Gradient-Based Learning Applied to Document Recognition*

### Key Innovation
LeNet-5 was the first successful CNN applied to practical problems — specifically handwritten digit recognition (MNIST). It demonstrated that end-to-end learning with gradient descent (backpropagation) could replace hand-engineered feature extractors.

### Architecture Summary

```
INPUT (32×32) → Conv1 (6×5×5, stride 1) → AvgPool (2×2, stride 2)
    → Conv2 (16×5×5, stride 1) → AvgPool (2×2, stride 2)
    → Conv3 (120×5×5) → FC (84) → FC (10, Softmax)
```

- **Input**: 32×32 grayscale image (larger than MNIST's 28×28 — padding was used).
- **Convolutions**: Three conv layers with 5×5 kernels. No padding — outputs shrink.
- **Pooling**: Subsampling layers using average pooling with learnable weights and sigmoid activation.
- **Activation**: Tanh / sigmoid throughout (ReLU was not yet popular).
- **Classifier**: Two fully connected layers + RBF (radial basis function) output.

### Performance Context
- 0.9% test error on MNIST — state-of-the-art at the time.
- Used commercially for check reading by US banks.

### Legacy
Established the template of `Conv → Pool → Conv → Pool → FC → FC` that dominated computer vision for a decade.

---

## AlexNet (2012)

**Authors**: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton  
**Paper**: *ImageNet Classification with Deep Convolutional Neural Networks*

### Key Innovations
1. **ReLU activation** — solved vanishing gradient, trained much faster than tanh/sigmoid.
2. **Dropout** — regularized fully connected layers to prevent overfitting.
3. **Data augmentation** — used random cropping, horizontal flips, and PCA color jitter.
4. **GPU training** — split the network across two GTX 580 GPUs.
5. **Local Response Normalization (LRN)** — a form of lateral inhibition (rarely used today).

### Architecture Summary

```
INPUT (224×224×3)
→ Conv1 (96×11×11, stride 4, pad 0) → LRN → MaxPool (3×3, stride 2)
→ Conv2 (256×5×5, pad 2, grouped) → LRN → MaxPool (3×3, stride 2)
→ Conv3 (384×3×3, pad 1)
→ Conv4 (384×3×3, pad 1)
→ Conv5 (256×3×3, pad 1) → MaxPool (3×3, stride 2)
→ FC (4096) → Dropout → FC (4096) → Dropout → FC (1000, Softmax)
```

### Performance Context
- Won ILSVRC 2012 with 15.3% top-5 error (second place was 26.2%).
- 60 million parameters, 650K neurons.

### Legacy
AlexNet triggered the deep learning revolution in computer vision. Every subsequent architecture builds on its core ideas.

---

## VGG (2014)

**Authors**: Karen Simonyan, Andrew Zisserman (Oxford VGG)  
**Paper**: *Very Deep Convolutional Networks for Large-Scale Image Recognition*

### Key Innovation
Systematically demonstrated that **depth matters**: stacking many 3×3 convolutions achieves larger effective receptive fields with fewer parameters than large kernels.

Two 3×3 conv layers have the same receptive field as one 5×5 layer but use only 18 weights instead of 25. Three 3×3 layers match a 7×7 kernel with 27 weights vs 49.

### Architecture Summary

| Block | VGG16 Layers | VGG19 Layers | Output Size |
|---|---|---|---|
| Block 1 | Conv(64) × 2, Pool | Conv(64) × 2, Pool | 112×112 |
| Block 2 | Conv(128) × 2, Pool | Conv(128) × 2, Pool | 56×56 |
| Block 3 | Conv(256) × 3, Pool | Conv(256) × 4, Pool | 28×28 |
| Block 4 | Conv(512) × 3, Pool | Conv(512) × 4, Pool | 14×14 |
| Block 5 | Conv(512) × 3, Pool | Conv(512) × 4, Pool | 7×7 |
| Head | FC(4096) × 2, FC(1000) | Same | 1×1 |

### Performance Context
- 7.3% top-5 error on ImageNet (ILSVRC 2014, 2nd place).
- 138 million parameters — mostly from the three FC layers.

### Legacy
VGG's simple, uniform design became the go-to architecture for transfer learning. Its main drawback is high computational cost.

---

## GoogLeNet / Inception (2014)

**Authors**: Christian Szegedy et al. (Google)  
**Paper**: *Going Deeper with Convolutions*

### Key Innovation
The **Inception module** applies convolutions of multiple sizes (1×1, 3×3, 5×5) and max pooling in parallel, concatenating their outputs. This lets the network learn features at different scales within the same layer.

1×1 convolutions are used for **bottleneck dimensionality reduction** — a 1×1 conv with fewer filters than input channels shrinks the channel count before expensive 3×3 or 5×5 convolutions.

### Inception Module (naive vs. with dimensionality reduction)

```
Naive:                              With 1×1 Bottleneck:
Input → 1×1 Conv ──────────────    Input → 1×1 Conv ──────────
      → 3×3 Conv ──────────────          → 1×1 Conv → 3×3 Conv
      → 5×5 Conv ──────────────          → 1×1 Conv → 5×5 Conv
      → 3×3 MaxPool ────────────          → 1×1 Proj → 3×3 MaxPool
      → Concatenate all                    → Concatenate all
```

### Architecture Summary
- 22 layers deep (27 with pooling).
- 9 Inception modules stacked with occasional max pooling.
- Auxiliary classifiers at intermediate layers for gradient flow during training.
- No fully connected layers — uses Global Average Pooling.
- 5 million parameters (12× fewer than AlexNet).

### Performance Context
- 6.7% top-5 error on ImageNet (ILSVRC 2014 winner).
- Much more parameter-efficient than VGG.

---

## ResNet (2015)

**Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft)  
**Paper**: *Deep Residual Learning for Image Recognition*

### Key Innovation
**Skip connections (residual connections)** solve the degradation problem — deeper networks had higher training error than shallower ones. Instead of learning an unreferenced mapping `H(x)`, ResNet learns the residual `F(x) = H(x) - x`, then computes `H(x) = F(x) + x`.

```
Plain Block:                       Residual Block:
Input → Conv → ReLU → Conv → ReLU  Input → Conv → ReLU → Conv → + → ReLU
                                                              ↑_______↓
                                                              (skip connection)
```

If the identity mapping is optimal, the block can simply learn zero weights. Skip connections also help gradient flow during backpropagation.

### Architecture Summary

| Layer Name | ResNet-50 | ResNet-101 | ResNet-152 |
|---|---|---|---|
| Conv1 | 7×7, 64, stride 2 | Same | Same |
| MaxPool | 3×3, stride 2 | Same | Same |
| Conv2_x | 3×Bottleneck(64) | 3×Bottleneck(64) | 3×Bottleneck(64) |
| Conv3_x | 4×Bottleneck(128) | 4×Bottleneck(128) | 8×Bottleneck(128) |
| Conv4_x | 6×Bottleneck(256) | 23×Bottleneck(256) | 36×Bottleneck(256) |
| Conv5_x | 3×Bottleneck(512) | 3×Bottleneck(512) | 3×Bottleneck(512) |
| Head | GAP, FC(1000) | Same | Same |

The bottleneck block reduces then expands dimensions: 1×1 (reduce) → 3×3 → 1×1 (expand).

### Performance Context
- 3.57% top-5 error on ImageNet (ILSVRC 2015 winner) — first result below human-level (5.1%).
- ResNet-152 has 60 million parameters but outperformed VGG-19 (138M) with much fewer.
- Skip connections are now a standard design pattern across deep learning (Transformers, UNet, GANs).

---

## DenseNet (2017)

**Authors**: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger  
**Paper**: *Densely Connected Convolutional Networks*

### Key Innovation
Each layer receives feature maps from **all preceding layers** as input, and passes its own feature maps to all subsequent layers. This creates *L(L+1)/2* connections for *L* layers.

```
Standard (ResNet):     x_l = H_l(x_{l-1}) + x_{l-1}
DenseNet:              x_l = H_l([x_0, x_1, ..., x_{l-1}])   ([...] = concatenation)
```

### Architecture Summary
- **Dense blocks**: Within each block, feature maps are concatenated. Spatial size is preserved.
- **Transition layers**: Between blocks — 1×1 conv (reduce channels) + 2×2 avg pool (reduce spatial size).
- **Growth rate (k)**: Each layer adds *k* new feature maps. Typical: k=12 or k=32. Total channels in a block = *k₀ + k × (L-1)*.

```
Input → Conv(2k) → DenseBlock × N → Transition → DenseBlock × N → ... → GAP → FC(Sotmax)
```

### Performance Context
- DenseNet-121 achieved comparable accuracy to ResNet-50 with half the parameters.
- Benefits: alleviates vanishing gradient, strengthens feature propagation, encourages feature reuse.

### Legacy
DenseNet showed extreme feature reuse works well, influencing later architectures like EfficientNet and modern transformer blocks.

---

## Architecture Comparison

| Model | Year | Depth | Parameters | ImageNet Top-5 Error | Key Innovation |
|---|---|---|---|---|---|
| LeNet-5 | 1998 | 7 | 60K | ~0.9% (MNIST) | First practical CNN |
| AlexNet | 2012 | 8 | 60M | 15.3% | ReLU, Dropout, GPU training |
| VGG-16 | 2014 | 16 | 138M | 7.3% | Deep 3×3 stacks |
| GoogLeNet | 2014 | 22 | 5M | 6.7% | Inception modules, 1×1 bottleneck |
| ResNet-50 | 2015 | 50 | 25.6M | 3.57% | Skip connections |
| DenseNet-121 | 2017 | 121 | 8M | ~3.7% | Dense connectivity |
