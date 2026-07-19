# Temporal Fusion & Attention

## The Motivation

RNNs and CNNs process sequences but give equal weight to every time step. In reality, certain time steps matter more — a sudden dip before a holiday, an anomalous sensor reading, or the beginning of a seasonal cycle. **Attention mechanisms** let the model learn *which* time steps are important for each prediction.

## Attention for Time Series

The core idea: at each decoding step, compute a weighted sum of encoder hidden states:

```
Attention score:   e_ti = score(h_t_dec, h_i_enc)
Attention weights: α_ti = softmax(e_ti)
Context vector:    c_t = Σ α_ti · h_i_enc
```

The score function can be:
- **Dot product**: `h_dec · h_enc` (simple, efficient)
- **Additive (Bahdanau)**: `v^T · tanh(W1·h_dec + W2·h_enc)` (more expressive)
- **Scaled dot product (Transformer)**: `(Q·K^T) / √d` (used in TFT)

**Why attention matters for time series:**
- It provides **interpretability** — the attention weights show which past steps influenced a prediction.
- It handles **variable-length context** — the model can look far back or focus on recent steps as needed.
- It mitigates the **long-term forgetting** problem of pure RNNs.

## Temporal Fusion Transformer (TFT)

The Temporal Fusion Transformer, introduced by Google in 2019, is a state-of-the-art architecture for multi-horizon forecasting. It combines LSTM, attention, and quantile regression in a unified framework.

### Architecture Components

#### 1. Variable Selection Network (VSN)

Real-world time series has many input features — some useful, some noise. The VSN applies a soft selection:

```python
# For each input feature, learn a gating weight via a GRN (Gated Residual Network)
feature_weights = softmax(GRN(inputs))
selected = feature_weights * inputs
```

This provides interpretability: you can inspect which features the model actually uses.

#### 2. LSTM Encoder-Decoder

Before the attention layer, TFT uses an LSTM-based encoder-decoder to process the selected features:

- **Encoder**: Reads historical inputs (lookback window).
- **Decoder**: Processes known future inputs (e.g., day-of-week for future dates) and generates initial predictions.

The LSTM handles **static covariates** (features that don't change over the horizon, like a store ID) and **known future inputs** (calendar features, known events).

#### 3. Interpretable Multi-Head Attention

TFT modifies the Transformer's multi-head attention:

- **Shared value weights** across heads — this forces all heads to attend to the same semantic features, making attention weights directly interpretable.
- Each head still has separate query/key projections, so it can focus on different temporal patterns.

The attention is applied over the encoder time steps, allowing the decoder to "look back" at relevant history.

#### 4. Quantile Outputs

Instead of a single point prediction, TFT outputs three quantiles per time step:

```python
# Default: p = [0.1, 0.5, 0.9]
output = Linear(hidden_dim, num_quantiles * horizon)
```

- **10th percentile**: Lower bound (pessimistic forecast)
- **50th percentile (median)**: Point forecast
- **90th percentile**: Upper bound (optimistic forecast)

This gives **prediction intervals** for uncertainty quantification — critical for business decisions (inventory, pricing, risk).

### Loss Function

TFT is trained with **quantile loss** (pinball loss):

```python
def quantile_loss(y_pred, y_true, q):
    error = y_true - y_pred
    return torch.max(q * error, (q - 1) * error)
```

The total loss is the average across all quantiles and time steps.

### TFT in Practice

The `pytorch-forecasting` library provides a well-tested TFT implementation:

```python
from pytorch_forecasting import TemporalFusionTransformer

tft = TemporalFusionTransformer.from_dataset(
    dataset,
    learning_rate=0.01,
    hidden_size=256,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=256,
    output_size=7,  # quantiles
)
```

## When to Use TFT vs Simpler Models

| Scenario | Recommended Model |
|---|---|
| Few features, short horizon | LSTM or TCN |
| Many features, uncertainty needs | TFT |
| Interpretability critical | TFT (VSN + attention weights) |
| Long sequences, large data | TFT or Transformer |
| Real-time / low latency | TCN (faster inference) |

## Self-Attention (Transformer) for Time Series

For very long sequences, pure Transformer architectures (without RNN) are gaining popularity:

- **TimeSeriesTransformer** (PyTorch): Uses self-attention on the full history, positional encoding, and causal masking.
- **Informer**: ProbSparse self-attention for efficient long-sequence handling.
- **Autoformer**: Replaces self-attention with auto-correlation for efficiency on long seasonal series.

The key limitation of pure attention: O(n²) memory for sequence length `n`, which becomes prohibitive for very long horizons. Efficient attention variants (ProbSparse, Linformer) address this.
