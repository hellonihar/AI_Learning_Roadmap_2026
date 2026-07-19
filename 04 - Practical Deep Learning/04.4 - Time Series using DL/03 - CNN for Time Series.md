# CNN for Time Series

## Why Use Convolutions on Sequences

Convolutional Neural Networks (CNNs) are not just for images. 1D convolutions slide a filter across a sequence, detecting local patterns — exactly what we need for time series. A convolution of kernel size `k` looks at `k` consecutive time steps:

```
Input:  [x0, x1, x2, x3, x4, x5]
Filter: [w0, w1, w2]
Output: [y0 = w0*x0 + w1*x1 + w2*x2,
         y1 = w0*x1 + w1*x2 + w2*x3, ...]
```

Each filter learns a *temporal pattern* — e.g., a sudden spike, a gradual slope, or a periodic shape. Multiple filters learn diverse patterns in parallel.

### Key Advantages

- **Parallelizable**: Unlike RNNs, convolutions process all time steps simultaneously — much faster training.
- **Hierarchical features**: Stacking convolutional layers builds progressively longer temporal receptive fields. Lower layers detect short patterns; higher layers combine them into complex behaviors.
- **Stable gradients**: No vanishing gradient across time — backpropagation flows through the network depth, not through time steps.

## 1D Convolution in PyTorch

```python
nn.Conv1d(in_channels=num_features, out_channels=num_filters,
          kernel_size=k, padding='same')
```

Note: PyTorch's `Conv1d` expects shape `(batch, channels, sequence_length)`. You need to permute or reshape time series data accordingly.

## Causal Convolutions

Standard convolutions look both forward and backward in time (they center the kernel on the current step). For forecasting, this creates **data leakage** — the model sees future values.

**Causal convolutions** fix this by ensuring that the output at time `t` depends only on inputs at times `t, t-1, t-2, ...`:

```
Standard (non-causal):  y_t = f(x_{t-1}, x_t, x_{t+1})   ← LEAKAGE
Causal:                 y_t = f(x_{t-k+1}, ..., x_t)      ← SAFE
```

In practice, causal convolution is achieved with **asymmetric padding**: pad the input on the left with `(kernel_size - 1)` zeros, so the kernel's rightmost element aligns with the current step.

```python
# Causal padding
pad = (kernel_size - 1)
x_padded = F.pad(x, (pad, 0))  # pad left side only
conv_out = conv(x_padded)
```

## WaveNet Architecture

WaveNet, originally designed for audio generation, is a powerful architecture for time series forecasting. Its key innovations:

### Dilated Convolutions

Dilated (or "atrous") convolutions insert gaps between kernel elements, exponentially expanding the receptive field without increasing parameters:

```
Layer 1: dilation=1, receptive field = 2
Layer 2: dilation=2, receptive field = 4
Layer 3: dilation=4, receptive field = 8
Layer 4: dilation=8, receptive field = 16
```

With `L` layers at dilations `[1, 2, 4, ..., 2^L]`, the receptive field is `2^L` — a stack of 10 layers sees 1024 time steps.

```python
nn.Conv1d(in_channels, out_channels, kernel_size=2,
          dilation=2**layer_idx, padding='same')
# Combined with causal padding
```

### Skip Connections

Each dilated convolution block outputs two paths:
1. A **residual connection** (added to the block's input) — stabilizes training in deep networks.
2. A **skip connection** — accumulated across all layers and fed to the final output layer.

```
input → conv(dilated) → tanh × sigmoid(gate) → residual path
                                           ↘ skip path → sum all skips → ReLU → conv → output
```

The gating mechanism (`tanh` and `sigmoid` activations multiplied element-wise) is borrowed from LSTM and controls information flow.

### WaveNet for Time Series

```python
class WaveNetBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=2,
                              dilation=dilation, padding=0)
        self.skip = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        # causal pad
        x_pad = F.pad(x, (self.conv.dilation[0], 0))
        out = self.conv(x_pad)
        gate = torch.tanh(out) * torch.sigmoid(out)
        residual = gate + x
        skip = self.skip(gate)
        return residual, skip
```

## CNN vs RNN for Time Series

| Aspect | CNN (TCN/WaveNet) | RNN (LSTM) |
|---|---|---|
| Training speed | Fast (parallel) | Slow (sequential) |
| Receptive field | Fixed (by depth/dilation) | Unlimited (hidden state) |
| Long sequences | Excellent (dilated convs) | Vanishing gradients |
| Memory | Less memory per step | More memory (hidden states) |
| Performance on TS | Often superior | Strong baseline |

**Temporal Convolutional Networks (TCN)** — a simpler variant using causal convolutions + dilations + residual blocks — are often the best "CNN for time series" starting point. WaveNet adds the gated activation for additional expressiveness.
