# Data Augmentation for Computer Vision

Data augmentation generates modified versions of training images to increase dataset diversity, reduce overfitting, and improve generalization. It is a critical component of modern vision pipelines — almost every state-of-the-art model uses aggressive augmentation.

## Geometric Transformations

These modify the spatial arrangement of pixels.

### Flip (Horizontal / Vertical)
- **Horizontal flip**: Mirror across the vertical axis. Standard for most tasks (objects are symmetric).
- **Vertical flip**: Mirror across horizontal axis. Use with caution (makes sense for aerial/satellite images, not for street scenes with gravity).

```
Original:  ┌──────┐        Horizontal Flip:  ┌──────┐
           │  ◄───┤        │  ───►  │
           └──────┘                          └──────┘
```

### Rotation
Rotate by 0°–360° (typical: ±10° for subtle variation, ±30° for aggressive augmentation). Rotation introduces empty corners that must be filled (zero-pad, reflect, or nearest-neighbor).

### Cropping and Rescaling
- **Random crop**: Extract a random patch and resize to original dimensions.
- **Random resized crop**: Crop at random scale (0.08–1.0 of original) and aspect ratio (3/4 to 4/3), then resize.
- Used extensively in training ResNet, EfficientNet, etc.

### Translation and Scaling
- **Translation**: Shift image by random offsets (fraction of width/height).
- **Scaling**: Zoom in/out by a random factor.

## Color Transformations

These modify pixel intensity values without changing spatial structure.

### Brightness
Multiply all pixel values by a factor (0.5–1.5). Darker or lighter variants.

### Contrast
Adjust contrast by scaling the difference from the mean intensity:
```
I' = (1 - α) · mean(I) + α · I    (α ∈ [0.5, 1.5])
```

### Saturation and Hue
- **Saturation**: Vary color intensity (grayscale ↔ vibrant).
- **Hue**: Shift color channels cyclically (e.g., +0.1 on a [0, 1] hue wheel).

### Color Jitter
Randomly adjust brightness, contrast, saturation, and hue together. Used in AlexNet, SimCLR, and most self-supervised learning pipelines.

## Advanced Augmentations

### CutOut / Random Erasing
Randomly mask out a square region of the image with zeros or random noise.

```
Original:  ┌───────┐         CutOut:   ┌───────┐
           │  Cat   │         │  Cat   │
           │        │         │  ████  │
           └───────┘         └───────┘
```

Forces the network to rely on context rather than a single discriminative feature.

### MixUp
Creates a convex combination of two images and their labels:

```
x' = λ · x₁ + (1 − λ) · x₂
y' = λ · y₁ + (1 − λ) · y₂    where λ ~ Beta(α, α), typically α = 0.2
```

Visually a blend: the network sees a transparent overlay of two images.

### CutMix
Combines CutOut and MixUp: replaces a region of one image with a patch from another, and mixes labels proportionally to the region area.

```
x' = mask ⊙ x₁ + (1 − mask) ⊙ x₂
y' = λ · y₁ + (1 − λ) · y₂    where λ = area ratio of the mask
```

CutMix consistently outperforms both CutOut and MixUp on ImageNet.

### RandAugment
A parameterized augmentation policy learned via search. Applies *N* randomly selected transformations from a set of *K* with magnitude *M*.

```
Steps:
1. Randomly select N transforms from {rotate, shearX, shearY, translateX,
   translateY, autoContrast, equalize, solarize, color, posterize, contrast,
   brightness, sharpness, etc.}
2. Apply each with magnitude M.
```

RandAugment achieved state-of-the-art results on ImageNet with only 2 hyperparameters (N and M), compared to AutoAugment's 30+.

## Implementation Notes

### Library Options

| Library | Best For | Notes |
|---|---|---|
| torchvision.transforms | PyTorch | Built-in, fast, easy to use |
| tf.keras.preprocessing | TensorFlow/Keras | Integrated, simple API |
| Albumentations | Performance | PyTorch/TF, fastest, most transforms |
| imgaug | Flexibility | Wide coverage, slower |
| Kornia | GPU pipelines | Differentiable augmentations for self-supervised learning |

### Augmentation Order

Recommended pipeline order:

1. **Geometric first** (random crop, flip, rotation)
2. **Color transforms** (brightness, contrast, hue)
3. **Advanced** (CutMix, MixUp — applied in the batch loop, not per-image)

### Training vs. Inference

- **Training**: Apply augmentation randomly (different versions each epoch).
- **Validation/Test**: Use deterministic preprocessing (center crop, no flip, no color jitter).

### Probability and Magnitude

- Apply each augmentation with probability 0.5 (common default).
- Start with mild magnitudes (e.g., rotation ±10°, brightness ±0.1).
- Increase magnitude adaptively or use built-in search (RandAugment).

## Summary

| Category | Transforms | When to Use |
|---|---|---|
| Geometric | Flip, rotate, crop, scale, translate | Almost always |
| Color | Brightness, contrast, saturation, hue | Improve lighting invariance |
| Advanced | CutOut, MixUp, CutMix, RandAugment | Small datasets, high overfitting risk |
