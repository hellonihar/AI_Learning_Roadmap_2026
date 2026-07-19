# CNN Architecture Components

A Convolutional Neural Network (CNN) is built from three core layer types: convolutional layers, pooling layers, and fully connected layers. These layers are stacked to form a hierarchical feature extractor.

## 1. Convolutional Layers

Conv layers are the backbone of a CNN. Each layer applies a set of learnable filters to the input, producing activation maps that highlight detected features.

### Key Properties

- **Local receptive field**: Each neuron sees only a small window of the previous layer.
- **Depth**: Number of filters determines output channel count.
- **Activation function**: Typically ReLU (Rectified Linear Unit) — `f(x) = max(0, x)` — applied element-wise after convolution to introduce non-linearity.

A typical conv layer:

```
Input → Conv2D (filters=64, kernel=3, stride=1, padding='same') → BatchNorm → ReLU
```

Standard conv layers use small kernels (3×3 or 5×3) and stack them deep to increase receptive field without excessive parameters.

## 2. Pooling Layers

Pooling reduces spatial dimensions, decreasing computation and providing translation invariance.

### Max Pooling

Takes the maximum value in each window:

```
Input (4×4):      2×2 Max Pooling (stride 2):   Output (2×2):
┌───┬───┬───┬───┐                               ┌───┬───┐
│ 1 │ 5 │ 3 │ 2 │       ┌───────┐               │ 8 │ 7 │
├───┼───┼───┼───┤       │ 1 5 │ 3 2 │           ├───┼───┤
│ 8 │ 3 │ 7 │ 1 │       │ 8 3 │ 7 1 │           │ 9 │ 5 │
├───┼───┼───┼───┤       └───────┘               └───┴───┘
│ 4 │ 6 │ 2 │ 9 │       max(1,5,8,3)=8
│ 9 │ 2 │ 5 │ 1 │       max(4,6,9,2)=9
└───┴───┴───┴───┘
```

### Average Pooling

Takes the mean instead of the maximum. Less aggressive than max pooling; useful when you want to preserve overall intensity.

### Global Average Pooling (GAP)

Reduces each feature map to a single scalar by averaging all spatial values. Used in modern architectures (ResNet, DenseNet) to replace fully connected layers, drastically reducing parameters and preventing overfitting.

| Pooling Type | Operation | Typical Use |
|---|---|---|
| Max Pooling | `max(window)` | Between conv blocks |
| Average Pooling | `mean(window)` | Between conv blocks |
| Global Avg Pooling | `mean(H, W)` per channel | Before classifier head |

## 3. Fully Connected Layers

After convolutional and pooling layers extract high-level features, fully connected (FC) layers act as the classifier.

- The final feature maps are flattened into a 1D vector.
- One or more FC layers project features to class scores.
- The last layer uses softmax (multi-class) or sigmoid (binary) activation.

Example classifier head:

```
Flatten() → Dense(4096, ReLU) → Dropout(0.5) → Dense(4096, ReLU) → Dropout(0.5) → Dense(num_classes, Softmax)
```

Modern CNNs use GAP + a single FC layer instead of multiple large FC layers to reduce overfitting.

## Feature Hierarchy

The layered structure of a CNN creates a natural hierarchy of learned features:

```
Layer 1 (Edge Detectors)
    └── Layer 2 (Textures / Patterns)
        └── Layer 3 (Parts / Motifs)
            └── Layer 4 (Object Parts)
                └── Layer 5 (Whole Objects)
```

### Visual Progression

| Stage | Layers | What is Detected | Example Visualization |
|---|---|---|---|
| Early (1-2) | Conv + Pool | Edges, color blobs, corners | Gabor-like oriented lines |
| Mid (3-5) | Conv + Pool | Textures, patterns, gradient bundles | Grid patterns, repeating textures |
| Late (6+) | Conv | Object parts | Wheels, eyes, windows |
| Final | FC / GAP | Complete objects | Dog face, car, building |

This hierarchy is empirically validated by Zeiler & Fergus (2014) in their visualization paper. Early layers are generic and transferable across tasks; later layers become task-specific.

## Typical Architecture Pattern

Most CNN architectures follow this blueprint:

```
INPUT → [Conv → ReLU → Pool?] × N → Classifier
```

Example: Simple CNN for CIFAR-10

| Layer | Type | Output Shape | Parameters |
|---|---|---|---|
| 0 | Input (32×32×3) | 32×32×3 | 0 |
| 1 | Conv 3×3, 32, ReLU, Same | 32×32×32 | 896 |
| 2 | Conv 3×3, 32, ReLU, Same | 32×32×32 | 9,248 |
| 3 | MaxPool 2×2 (stride 2) | 16×16×32 | 0 |
| 4 | Conv 3×3, 64, ReLU, Same | 16×16×64 | 18,496 |
| 5 | Conv 3×3, 64, ReLU, Same | 16×16×64 | 36,928 |
| 6 | MaxPool 2×2 (stride 2) | 8×8×64 | 0 |
| 7 | Global Avg Pooling | 64 | 0 |
| 8 | Dense, 10, Softmax | 10 | 650 |

Total ≈ 66K parameters — orders of magnitude fewer than a fully connected network of similar capacity.
