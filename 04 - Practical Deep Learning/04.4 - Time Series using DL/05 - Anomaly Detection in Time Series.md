# Anomaly Detection in Time Series

## The Problem

Anomaly detection in time series identifies time steps where the system behaves unexpectedly — a server CPU spike, a fraudulent transaction, a sensor failure. Deep learning approaches learn the "normal" patterns from historical data and flag deviations.

There are three main categories of anomalies:
- **Point anomalies**: A single step that deviates (e.g., a sudden sensor glitch).
- **Contextual anomalies**: A value that is normal in isolation but abnormal for its temporal context (e.g., 30°C in December).
- **Collective anomalies**: A sequence of steps that is abnormal as a whole (e.g., an entire daily pattern shifted upward).

## Reconstruction-Based (Autoencoder)

The autoencoder approach: compress the input window into a low-dimensional latent space (encoder) and reconstruct it (decoder). If the reconstruction error is high, the window is anomalous.

### Architecture

```
Input:  [x_1, x_2, ..., x_T]
Encoder:  Linear or 1D-CNN → bottleneck → Decoder: 1D-CNN or Linear
Output: [x̂_1, x̂_2, ..., x̂_T]
Anomaly score: MSE(x, x̂)
```

### Training

- Train on **normal-only** data (or a majority of normal data with robust loss).
- After training, compute reconstruction error on all data.
- Threshold: set at e.g. mean + 3× std of training reconstruction errors.

```python
class ConvAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, n_features, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Crop or interpolate to match input length if needed
        return decoded
```

**Pros**: Unsupervised — no labeled anomalies needed. Simple to implement.  
**Cons**: May reconstruct anomalies too well if trained on mixed data. Sensitive to window size.

## Prediction-Based (LSTM Forecasting Error)

Train a forecasting model (e.g., LSTM) to predict the next step. If the actual value differs significantly from the prediction, it's anomalous.

### How It Works

1. Train LSTM to forecast `y_{t+1}` from `[x_{t-w+1}, ..., x_t]`.
2. On new data, compute prediction error: `e_t = |y_t - ŷ_t|`.
3. Flag as anomaly if `e_t > threshold`.

```python
# During inference
model.eval()
errors = []
with torch.no_grad():
    for batch in test_loader:
        X, y = batch
        y_pred = model(X)
        errors.append(torch.abs(y - y_pred).cpu().numpy())

errors = np.concatenate(errors)
# Threshold: e.g., 95th percentile
threshold = np.percentile(errors, 95)
anomalies = errors > threshold
```

**Dynamic thresholding**: Rather than a static threshold, use a rolling window (e.g., threshold = rolling_mean + 3× rolling_std). This adapts to changes in volatility.

**Pros**: Simple, interpretable (you see why — forecast vs actual).  
**Cons**: Requires labeled normal data for training. Error accumulates in multi-step settings.

## LSTM-Autoencoder for Anomaly Detection

Combines both approaches: use an LSTM encoder-decoder to reconstruct entire sequences, leveraging the LSTM's ability to model temporal dynamics.

```
LSTM Encoder:    input sequence → final hidden state (h_T)
Repeat vector:   h_T repeated T times
LSTM Decoder:    repeated h_T → reconstructed sequence
Reconstruction:  MSE over the full window
```

```python
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # Encode
        _, (h_n, c_n) = self.encoder(x)
        # Repeat context for each decoder step
        context = h_n[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        # Decode
        decoder_out, _ = self.decoder(context, (h_n, c_n))
        return self.fc(decoder_out)
```

The anomaly score for a window is the mean reconstruction error across all time steps in the window. Windows with high error contain anomalous points.

## Evaluation: Precision and Recall on Anomalies

Anomaly detection is a highly imbalanced classification problem — anomalies are rare.

### Metrics

| Metric | Formula | Meaning |
|---|---|---|
| Precision | TP / (TP + FP) | Of the points flagged, how many are real anomalies? |
| Recall (TPR) | TP / (TP + FN) | Of the real anomalies, how many did we catch? |
| F1 Score | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |
| FPR | FP / (FP + TN) | False alarms among normal points |

### Challenges

- **Point-adjustment protocol**: If a model flags anomaly at step `t+2` but the real anomaly starts at `t`, many papers count it as a correct detection within an "event region" (contiguous anomaly block). This is standard practice but inflates metrics.
- **Threshold selection**: The threshold is a trade-off — lower threshold catches more anomalies (higher recall) but triggers more false alarms (lower precision). The F1 score at various thresholds gives a complete picture.

### Comparison Summary

| Method | Supervision | Best For |
|---|---|---|
| Autoencoder (conv) | Unsupervised | Pattern-based anomalies, sensor data |
| LSTM forecasting error | Semi-supervised | Point anomalies, real-time detection |
| LSTM-Autoencoder | Unsupervised | Complex temporal anomalies, long windows |
| TFT quantile outputs | Supervised | When labeled anomalies exist, probabilistic detection |

The choice depends on your data and whether you have labeled anomalies. In practice, try the LSTM forecasting error first — it's simple, fast, and interpretable.
