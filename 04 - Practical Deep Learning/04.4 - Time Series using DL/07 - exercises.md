# Exercises — Time Series using Deep Learning

## Exercise 1: Create Sliding Windows

Given the sequence `[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]`, write code (or manually compute) to create sliding windows with `window_size=3` and `horizon=2`. How many samples do you get?

<details>
<summary>Answer</summary>

```python
import numpy as np

data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
window_size = 3
horizon = 2

X, y = [], []
for i in range(len(data) - window_size - horizon + 1):
    X.append(data[i : i + window_size])
    y.append(data[i + window_size : i + window_size + horizon])

print("X:", np.array(X))
print("y:", np.array(y))
print(f"Number of samples: {len(X)}")
```

```
X: [[ 10  20  30]
    [ 20  30  40]
    [ 30  40  50]
    [ 40  50  60]
    [ 50  60  70]
    [ 60  70  80]]
y: [[40 50]
    [50 60]
    [60 70]
    [70 80]
    [80 90]
    [90 100]]
Samples: 6
```

Formula: `len(data) - window_size - horizon + 1 = 10 - 3 - 2 + 1 = 6`
</details>

## Exercise 2: Stationarity Check

You have a time series with a clear upward trend and increasing variance over time. Name two transformations that can help make it stationary.

<details>
<summary>Answer</summary>

1. **Log transformation** — stabilizes increasing variance (converts multiplicative patterns to additive).
2. **First-order differencing** (`y_t - y_{t-1}`) — removes linear trend.

Apply log first, then difference if needed: `log(y_t) - log(y_{t-1})`.
</details>

## Exercise 3: Causal Convolution Explanation

Explain in 2–3 sentences what a causal convolution is and why it is necessary for time series forecasting. Contrast it with a standard 1D convolution.

<details>
<summary>Answer</summary>

A causal convolution ensures the output at time `t` depends only on inputs at times `t, t-1, t-2, ...` (past and present), never on future values. This is achieved by left-padding the input before applying the convolution kernel. Standard 1D convolutions are centered or right-padded, which allows the kernel to "look ahead" — causing data leakage from future time steps into predictions. For forecasting, this leakage would give the model access to information it wouldn't have at inference time, producing falsely optimistic performance.
</details>

## Exercise 4: Compare RNN vs CNN for Time Series

Fill in the table below by marking which architecture is better for each criterion:

| Criterion | Better Choice (RNN/LSTM or CNN) |
|---|---|
| Training speed | |
| Long-range dependencies | |
| Parallelization | |
| Vanishing gradient robustness | |
| Variable-length input | |

<details>
<summary>Answer</summary>

| Criterion | Better Choice |
|---|---|
| Training speed | **CNN** (parallel convolutions) |
| Long-range dependencies | **RNN/LSTM** (theoretically unlimited via hidden state, though in practice CNNs with dilations can match this) |
| Parallelization | **CNN** (convolutions over all steps in parallel) |
| Vanishing gradient robustness | **CNN** (no BPTT through time; gradients flow through network depth only) |
| Variable-length input | **RNN** (inherently handles variable length; CNNs need fixed input size or adaptive pooling) |
</details>

## Exercise 5: Implement Anomaly Detection Threshold

Given an array of prediction errors `[0.1, 0.2, 0.05, 0.3, 1.5, 0.15, 0.08, 2.1, 0.12, 0.25]`, implement code to:
1. Compute a threshold at the 90th percentile.
2. Identify which points are anomalies.
3. Compute the threshold as `mean + 2 * std`.
Compare the results.

<details>
<summary>Answer</summary>

```python
import numpy as np

errors = np.array([0.1, 0.2, 0.05, 0.3, 1.5, 0.15, 0.08, 2.1, 0.12, 0.25])

# Method 1: Percentile
threshold_pct = np.percentile(errors, 90)
anomalies_pct = errors > threshold_pct
print(f"Threshold (90th percentile): {threshold_pct:.4f}")
print(f"Anomalies: {errors[anomalies_pct]}")

# Method 2: Mean + 2*std
threshold_std = errors.mean() + 2 * errors.std()
anomalies_std = errors > threshold_std
print(f"Threshold (mean + 2*std): {threshold_std:.4f}")
print(f"Anomalies: {errors[anomalies_std]}")
```

Output:
- Percentile (90th): threshold = 1.35, anomalies = [1.5, 2.1]
- Mean + 2σ: threshold = 1.62, anomalies = [2.1]

The percentile method is more aggressive (catches more). The std method is more conservative.
</details>

## Exercise 6: Multi-Step Strategies Comparison

List the three multi-step forecasting strategies. For each, give one advantage and one disadvantage.

<details>
<summary>Answer</summary>

1. **Direct (Multi-Output)**: Train one model that outputs all H future steps at once.
   - Advantage: No error accumulation across steps.
   - Disadvantage: Horizon is fixed; model cannot adapt to different horizons at inference.

2. **Recursive (Iterated)**: Train a one-step model and feed predictions back as input.
   - Advantage: Single model works for any horizon.
   - Disadvantage: Errors compound — a bad step-1 prediction corrupts steps 2 through H.

3. **Direct-Recursive Hybrid**: Combine both approaches (e.g., direct outputs as features for a recursive model, or ensemble average).
   - Advantage: Reduces bias (from direct) and variance (from recursive).
   - Disadvantage: More complex training pipeline; two models to maintain and tune.
</details>

## Exercise 7: LSTM Seq2Seq Concept

Draw (or describe) the data flow through a sequence-to-sequence LSTM for time series forecasting with:
- Encoder input: 24 time steps
- Decoder output: 6 future steps
- Hidden size: 64
- Features: 1 (univariate)

Identify where the encoder hidden state is handed off to the decoder.

<details>
<summary>Answer</summary>

Data flow:
1. **Encoder**: Input shape `(batch, 24, 1)` → LSTM produces output `(batch, 24, 64)` and final states `h_n: (1, batch, 64)`, `c_n: (1, batch, 64)`.
2. **Context vector**: The final encoder hidden state `h_n` (shape `(1, batch, 64)`) is the **only connection** between encoder and decoder.
3. **Decoder initialization**: The decoder LSTM is initialized with `h_n` and `c_n` from the encoder.
4. **Decoder step 1**: Input is the last value of the encoder window (or a start token) → decoder produces first prediction.
5. **Decoder step 2**: Input is step 1's prediction (or teacher-forced true value) → decoder produces second prediction, using its own updated `h`, `c`.
6. Repeat for all 6 steps, collecting each prediction into the output `(batch, 6, 1)`.

The hand-off point is the **encoder's final hidden state** being passed as the decoder's initial hidden state.
</details>

## Exercise 8: TFT Components

Name four key components of the Temporal Fusion Transformer architecture. For each, explain its purpose in one sentence.

<details>
<summary>Answer</summary>

1. **Variable Selection Network**: Applies soft feature selection to determine which input features are most relevant at each time step, providing interpretability.
2. **LSTM Encoder-Decoder**: Processes historical inputs and known future inputs, capturing sequential dependencies before the attention layer.
3. **Interpretable Multi-Head Attention**: Allows the decoder to attend to relevant encoder time steps, with shared value weights so attention heads remain interpretable.
4. **Quantile Outputs**: Predicts multiple quantiles (e.g., 10th, 50th, 90th percentiles) per time step to produce prediction intervals for uncertainty quantification.
</details>
