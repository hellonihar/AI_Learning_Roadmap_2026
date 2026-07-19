# Convolution Operation

## What is Convolution?

In the context of computer vision, convolution is a mathematical operation that slides a small matrix (called a kernel or filter) over an input image, computing element-wise multiplications and summing the results to produce a feature map. This operation detects local patterns such as edges, textures, and shapes.

### Discrete 2D Convolution Formula

For an input image **I** of size *H × W* and a kernel **K** of size *k₁ × k₂*, the discrete 2D convolution at position *(i, j)* is:

```
S(i, j) = (I * K)(i, j) = Σₘ Σₙ I(i + m, j + n) · K(m, n)
```

where the sums run over the kernel dimensions *m = 0..k₁-1* and *n = 0..k₂-1*.

### Text Diagram: 3×3 Kernel on 5×5 Input

```
Input (5×5)          Kernel (3×3)        Output (3×3)
┌─────────────┐      ┌─────────┐         ┌─────────────┐
│ 1  2  3  4  5│      │ 1  0  -1│         │ -4  -4   0  │
│ 6  7  8  9 10│      │ 2  0  -2│         │ -4  -4   0  │
│11 12 13 14 15│      │ 1  0  -1│         │ -4  -4   0  │
│16 17 18 19 20│      └─────────┘         └─────────────┘
│21 22 23 24 25│
└─────────────┘

Top-left output = 1·1 + 2·0 + 3·(-1) + 6·2 + 7·0 + 8·(-2) + 11·1 + 12·0 + 13·(-1)
                = 1 + 0 - 3 + 12 + 0 - 16 + 11 + 0 - 13 = -4
```

## Kernels / Filters

A kernel is a small weight matrix that the network learns during training. Each filter detects a specific feature. Common hand-designed filters include:

| Kernel Type       | 3×3 Example          | Detects          |
|-------------------|----------------------|------------------|
| Vertical Edge     | `[[-1,0,1],[-2,0,2],[-1,0,1]]` (Sobel) | Vertical edges   |
| Horizontal Edge   | `[[-1,-2,-1],[0,0,0],[1,2,1]]`        | Horizontal edges |
| Blur              | `[[1,1,1],[1,1,1],[1,1,1]]` / 9       | Averaging        |

In a CNN, all kernel values are initialized randomly and learned via backpropagation.

## Stride

Stride controls how many pixels the kernel shifts at each step.

- **Stride = 1**: kernel moves one pixel at a time — dense, overlapping windows.
- **Stride = 2**: kernel jumps two pixels — smaller output, reduced computation.

### Output Size Formula

Given input size *W*, kernel size *K*, stride *S*, and padding *P*:

```
Output size = ⌊(W - K + 2P) / S⌋ + 1
```

## Padding

Padding adds extra pixels (usually zeros) around the input border.

- **Valid Padding (P = 0)**: No padding. Output shrinks.  
  `Output = W - K + 1` (for S = 1).

- **Same Padding**: Pad so output size equals input size (when S = 1).  
  `P = (K - 1) / 2` (requires K odd).

```
Example: W = 32, K = 3, S = 1
  Valid:  Output = 32 - 3 + 1 = 30
  Same:   P = 1 → Output = (32 - 3 + 2) / 1 + 1 = 32
```

## Channel Handling (RGB Input)

Real images have multiple channels (R, G, B for color). A 2D convolution extends to 3D:

- Input shape: `H × W × C_in` (e.g., 32×32×3 for RGB)
- Each filter has shape `K × K × C_in`
- The filter slides across the spatial dimensions, computing the sum across all channels at each position.

### Multiple Filters

A convolutional layer typically uses *N* filters, each producing one output channel:

```
Input:  H × W × C_in
Kernel: K × K × C_in   (one per filter)
Output: H_out × W_out × N   (stack N feature maps)
```

### Text Diagram: Multi-channel Convolution

```
Input (H×W×3)         Filter 1 (K×K×3)     Output Map 1
┌─────┐ ┌─────┐ ┌─────┐  ┌─────────┐       ┌───────────┐
│  R  │ │  G  │ │  B  │  │ R│G│B  │       │ Feature   │
│     │ │     │ │     │  │ ─┼─┼─  │       │ Map 1     │
└─────┘ └─────┘ └─────┘  │  3D kernel│       └───────────┘
                          └─────────┘       
                                       
Filter 2 (K×K×3) → Output Map 2
Filter 3 (K×K×3) → Output Map 3
...
Filter N (K×K×3) → Output Map N
```

Each output map captures a different pattern (e.g., one detects horizontal edges, another detects red-green boundaries).

## Bias Term

After convolution, a learnable bias is added per output channel:

```
Output[i, j, k] = Σₘ Σₙ Σ_c Input[i+m, j+n, c] · Kernel[m, n, c, k] + Bias[k]
```

## Why Convolution?

Convolution provides three key advantages over fully connected layers:

1. **Sparse interactions**: Each output neuron connects to only a local region (the receptive field), not every input pixel.
2. **Parameter sharing**: The same kernel weights are reused across all spatial positions, drastically reducing parameters.
3. **Equivariance to translation**: If an object shifts in the input, its feature map shifts correspondingly.

A fully connected layer on a 32×32×3 image would require 3072 weights per output neuron. A 3×3 conv filter with 3 input channels uses only 27 weights — a 100× reduction.
