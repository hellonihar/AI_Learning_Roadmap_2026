# Exercises: CNN & Transfer Learning

## Exercise 1: Output Size Calculation

Compute the output spatial dimensions for the following convolution:

- Input: 32 × 32 × 3
- Kernel: 5 × 5
- Padding: 2
- Stride: 1
- Number of filters: 64

**Questions:**
a) What is the output height and width?
b) What is the output depth?
c) How many parameters are in this layer (including bias)?

---

## Exercise 2: Stride and Padding Effects

Given a 128 × 128 grayscale input, compute the output size for:

a) Kernel 3×3, pad=0, stride=1
b) Kernel 3×3, pad=1, stride=2
c) Kernel 7×7, pad=3, stride=2
d) Kernel 1×1, pad=0, stride=1

---

## Exercise 3: Implement 2D Convolution with NumPy

Implement a `conv2d(input, kernel, padding, stride)` function similar to the code walkthrough, but add support for **dilation**.

Convolution with dilation rate *d*: the kernel elements are spaced *d* apart.

```
Kernel (3×3, d=2): effective size = 3 + (3-1)*(d-1) = 5
Positions: (0,0), (0,2), (0,4), (2,0), (2,2), (2,4), (4,0), (4,2), (4,4)
```

**Task**: Extend the `conv2d_single` function to accept a `dilation` parameter.

---

## Exercise 4: Explain Why ResNet Works

In your own words, explain how skip connections (residual connections) solve the degradation problem. Include:

- What the degradation problem is (be specific — it's not overfitting or vanishing gradient).
- How the identity mapping `H(x) = F(x) + x` changes the learning target.
- Why this makes it easier to train deeper networks.

---

## Exercise 5: Transfer Learning Decision

You are given the following scenarios. For each, recommend feature extraction or fine-tuning and justify:

a) Classifying 10 types of dog breeds using 50 images per breed. Pre-trained model: ImageNet.

b) Classifying lung nodules in CT scans (2000 scans, each showing a 256×256 region). Pre-trained model: ImageNet.

c) Classifying hand-drawn sketches of everyday objects (100 sketches per class, 20 classes). Pre-trained model: ImageNet.

d) Classifying satellite images of urban vs. rural areas (10,000 images per class). Pre-trained model: ResNet-50 on ImageNet.

---

## Exercise 6: Pooling Comparison

Compare max pooling and average pooling. For each of the following tasks, state which pooling type is preferred and why:

a) Detecting whether a face is present anywhere in an image.
b) Estimating the average brightness of an image region.
c) Texture classification where fine detail matters.

---

## Exercise 7: Compute Parameter Count for VGG-Style Block

Given a VGG-style block:

```
Conv2D(64, 3×3, pad='same') → ReLU → Conv2D(64, 3×3, pad='same') → ReLU → MaxPool(2×2, stride=2)
```

Input: 224 × 224 × 3

a) How many parameters does this block have?
b) What is the output shape after the block?
c) How many FLOPs (multiply-adds) does the first convolution perform?

---

## Exercise 8: Data Augmentation Design

You are training a model to classify traffic signs from photos taken by dashboard cameras. Propose an augmentation strategy. For each augmentation:

- State the transformation and its parameters.
- Explain why it helps (what invariance it provides).
- Identify any augmentations that would be harmful.

---

# Answers

## Answer 1

Output size = ⌊(W - K + 2P) / S⌋ + 1 = ⌊(32 - 5 + 4) / 1⌋ + 1 = 32

a) Height = 32, Width = 32
b) Depth = 64 (number of filters)
c) Parameters per filter = 5 × 5 × 3 = 75. Total conv params = 75 × 64 = 4800. Bias = 64. **Total = 4864**.

## Answer 2

a) (128 - 3 + 0) / 1 + 1 = 126
b) (128 - 3 + 2) / 2 + 1 = 64
c) (128 - 7 + 6) / 2 + 1 = 64
d) (128 - 1 + 0) / 1 + 1 = 128

## Answer 3

```python
def conv2d_dilated(input_2d, kernel, pad=0, stride=1, dilation=1):
    H, W = input_2d.shape
    k = kernel.shape[0]
    eff_k = k + (k - 1) * (dilation - 1)

    if pad > 0:
        padded = np.pad(input_2d, pad, mode='constant')
    else:
        padded = input_2d

    H_out = (padded.shape[0] - eff_k) // stride + 1
    W_out = (padded.shape[1] - eff_k) // stride + 1
    output = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            i_start = i * stride
            j_start = j * stride
            patch = padded[i_start:i_start + eff_k:dilation,
                           j_start:j_start + eff_k:dilation]
            output[i, j] = np.sum(patch * kernel)

    return output
```

## Answer 4

The degradation problem is that adding more layers to a network increases **training error** (not just test error), meaning deeper models are harder to optimize than shallower ones. This is distinct from vanishing gradients (which affect convergence speed) and overfitting (which affects generalization).

ResNet introduces skip connections so each block learns the residual `F(x) = H(x) - x` rather than the full mapping `H(x)`. The block output is `F(x) + x`. If the identity `H(x) = x` is optimal, the block can set `F(x) ≈ 0` (easy — just learn small weights). Without skip connections, a stack of nonlinear layers must learn the identity mapping directly, which is difficult.

Skip connections also provide a direct gradient highway from the loss to early layers, mitigating vanishing gradients.

## Answer 5

a) **Feature extraction**. Very small dataset, and the task is similar to ImageNet. Training the full network would overfit severely.

b) **Fine-tuning**. Medical images differ significantly from natural images. With 2000 scans, the dataset is large enough to adapt features.

c) **Fine-tune last 1/3 or 1/2 layers**. Sketches share low-level features (edges) with natural images but differ in high-level appearance. Medium dataset size.

d) **Fine-tune all (low LR)**. Large dataset, domain is moderately different. End-to-end fine-tuning will adapt features like satellite-specific textures.

## Answer 6

a) **Max pooling**. Detecting presence benefits from the strongest activation, regardless of exact location.

b) **Average pooling**. Estimating brightness requires averaging pixel values, which max pooling would bias toward the brightest pixel.

c) **Average pooling** (or no pooling). Max pooling discards most spatial information, which can hurt fine-grained texture discrimination. Some texture models use average pooling or strided convolutions instead.

## Answer 7

a) First conv: 3 × 3 × 3 × 64 + 64 = 1728 + 64 = 1792.  
   Second conv: 3 × 3 × 64 × 64 + 64 = 36864 + 64 = 36928.  
   **Total = 1792 + 36928 = 38720 parameters** (no parameters in pool).

b) After Conv1: 224 × 224 × 64 (same padding). After Conv2: 224 × 224 × 64.  
   After MaxPool(2,2): **112 × 112 × 64**.

c) FLOPs per conv: kernel_size² × C_in × C_out × H_out × W_out  
   = 9 × 3 × 64 × 224 × 224 = **86,704,128** multiply-adds.

## Answer 8

**Recommended augmentations:**
- **Random rotation (±15°)**: Handles camera tilt on vehicles.
- **Random brightness/contrast**: Handles varying lighting (sun, shadow, tunnel).
- **Gaussian blur**: Handles motion blur.
- **Random crop + resize**: Handles varying distances to signs.

**Harmful augmentations:**
- **Horizontal flip**: Would flip "LEFT" and "RIGHT" arrow signs, creating label noise.
- **Hue shift (large)**: Could change red stop signs to unnatural colors.
- **Vertical flip**: Signs would appear upside-down (unrealistic for dashboard cameras).
