# Time Series using Deep Learning — Cheatsheet

## Time Series Concepts

| Concept | Description |
|---|---|
| **Trend** | Long-term direction (up/down/flat) |
| **Seasonality** | Repeating pattern at fixed intervals |
| **Noise** | Random variation after removing trend & seasonality |
| **Stationarity** | Constant mean, variance, autocorrelation over time |
| **Autocorrelation** | Correlation of a series with its lagged values |
| **Lag features** | Past values of the target used as predictors |

## Stationarity Checks & Fixes

| Method | Purpose |
|---|---|
| **ADF test** | p < 0.05 → stationary |
| **Differencing** | `y_t - y_{t-1}` removes linear trend |
| **Log transform** | Stabilizes multiplicative variance |
| **Seasonal differencing** | `y_t - y_{t-m}` removes seasonality |

## Windowing (Sliding Window)

```
Window size (lookback) = W, Horizon (forecast) = H
X shape: (num_samples, W, num_features)
y shape: (num_samples, H, 1) or (num_samples, H)
Samples = len(data) - W - H + 1
```

**Always split chronologically — never shuffle.**

## Architectures

| Model | Key Idea | Best For |
|---|---|---|
| **LSTM** | Gated cell + hidden state | Medium-length sequences, baseline |
| **Seq2Seq LSTM** | Encoder → hidden state → decoder | Multi-step forecasting |
| **1D CNN / TCN** | Causal dilated convolutions | Fast training, long sequences |
| **WaveNet** | Gated dilated convs + skip connections | Very long sequences, audio |
| **TFT** | VSN + LSTM + Attention + Quantiles | Multivariate, interpretability, uncertainty |
| **Transformer** | Self-attention, positional encoding | Very large datasets, long sequences |

## Causal Convolution

```
Pad left with (kernel_size - 1) * dilation zeros
Output at t depends only on x_t, x_{t-1}, ..., x_{t-k+1}
```

```python
nn.functional.pad(x, (pad, 0))  # left-only padding
```

## Evaluation Metrics

| Metric | Formula | Range | Notes |
|---|---|---|---|
| **MAE** | mean(\|y - ŷ\|) | [0, ∞) | Units match target |
| **RMSE** | sqrt(mean((y - ŷ)²)) | [0, ∞) | Penalizes large errors |
| **MAPE** | mean(\|(y - ŷ)/y\| × 100) | [0, ∞) | Scale-independent (%); undefined at y=0 |
| **sMAPE** | mean(2\|y - ŷ\|/(\|y\|+\|ŷ\|) × 100) | [0, 200] | Symmetric; handles zeros better |

```python
def mae(y, y_hat): return np.mean(np.abs(y - y_hat))
def rmse(y, y_hat): return np.sqrt(np.mean((y - y_hat)**2))
def mape(y, y_hat): return np.mean(np.abs((y - y_hat) / y)) * 100
def smape(y, y_hat): return np.mean(2 * np.abs(y - y_hat) / (np.abs(y) + np.abs(y_hat))) * 100
```

## Multi-Step Strategies

| Strategy | Error Propagation | Horizon |
|---|---|---|
| **Direct** | No accumulation | Fixed |
| **Recursive** | Compounds | Any |
| **Hybrid** | Reduced | Any |

## Anomaly Detection

| Method | Approach | Supervision |
|---|---|---|
| **Autoencoder** | Reconstruct window; high error = anomaly | Unsupervised |
| **LSTM forecast error** | \|y - ŷ\| > threshold | Semi-supervised |
| **LSTM-Autoencoder** | Encode/decode with LSTM; reconstruction error | Unsupervised |

### Anomaly Metrics

| Metric | Formula |
|---|---|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2 × P × R / (P + R) |

## Common Hyperparameters

| Parameter | Typical Range |
|---|---|
| Window size | 1–2 seasonal cycles |
| Hidden size (LSTM) | 32–256 |
| Number of LSTM layers | 1–3 |
| Kernel size (CNN) | 3–7 |
| Dilation rate | 2^layer (TCN/WaveNet) |
| Learning rate | 1e-4 – 1e-2 |
| Batch size | 32–256 |
| Gradient clipping | 0.5–5.0 |
| Teacher forcing ratio | Anneal 1.0 → 0.0 |

## PyTorch Quick Reference

```python
# LSTM
nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

# 1D Conv (permute input to (batch, channels, seq_len))
nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')

# Causal Conv
nn.functional.pad(x, ((kernel_size-1)*dilation, 0))

# Dataset
class WindowDataset(Dataset):
    def __getitem__(self, idx):
        return self.data[idx:idx+W], self.data[idx+W:idx+W+H]

# No shuffle for train_loader
DataLoader(..., shuffle=False)

# Quantile loss
def quantile_loss(y_pred, y_true, q):
    err = y_true - y_pred
    return torch.max(q * err, (q-1) * err)
```
