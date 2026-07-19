# RNN / LSTM for Time Series

## Why RNNs for Time Series

Recurrent Neural Networks (RNNs) are the natural fit for sequential data. They maintain a hidden state that carries information across time steps, allowing them to model temporal dependencies of arbitrary length. For time series, each step `t` corresponds to a single observation (or a window step), and the hidden state compresses the history of all prior steps.

However, vanilla RNNs suffer from the **vanishing gradient problem** — gradients shrink exponentially over long sequences, making it impossible to learn long-range dependencies (e.g., annual seasonality in monthly data).

## LSTM to the Rescue

Long Short-Term Memory (LSTM) networks solve this with a gated cell structure:

- **Forget gate**: Decides what old information to discard.
- **Input gate**: Decides what new information to store.
- **Cell state**: The "conveyor belt" that carries information across many steps with minimal transformation — this is what preserves long-term dependencies.
- **Output gate**: Decides what to output from the current cell state.

The LSTM cell takes `(h_{t-1}, c_{t-1}, x_t)` and produces `(h_t, c_t)`. The hidden state `h_t` is used for predictions; the cell state `c_t` carries long-term memory.

For time series, a typical LSTM layer looks like:

```python
nn.LSTM(input_size=num_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
```

## Sequence-to-Sequence Forecasting

For multi-step forecasting (predicting values at `t+1, t+2, ..., t+H`), the seq2seq architecture is widely used:

### Encoder
The encoder reads the input window `[x_1, x_2, ..., x_T]` and produces a final hidden state `h_T` and cell state `c_T`. These encode the entire input sequence.

```python
encoder = nn.LSTM(input_size, encoder_hidden, batch_first=True)
encoder_out, (h_n, c_n) = encoder(X)  # use h_n, c_n as context
```

### Decoder
The decoder starts with the encoder's final states and generates predictions one step at a time. At each step, it takes its previous hidden state and the previous predicted value (or ground truth during teacher forcing):

```python
decoder = nn.LSTM(output_size, decoder_hidden, batch_first=True)
decoder_input = X[:, -1:, :]  # last input value to start
for t in range(horizon):
    decoder_out, (h_n, c_n) = decoder(decoder_input, (h_n, c_n))
    predictions.append(decoder_out)
    decoder_input = decoder_out  # feed predicted value as next input
```

During training, **teacher forcing** feeds the true value `y_t` instead of the predicted `ŷ_t` at each step, which stabilizes and accelerates training. The teacher forcing ratio is typically annealed from 1.0 to 0.0 over training epochs.

## Multi-Step Ahead Strategies

When forecasting multiple steps into the future, three strategies are common:

### 1. Direct (Multi-Output) Strategy

Train a single model that outputs a vector of length `H` (the horizon):

```python
nn.LSTM(input_size, hidden_dim, batch_first=True)
nn.Linear(hidden_dim, horizon)  # output all steps at once
```

**Pros**: Each forecast step uses its own learned parameters; no error accumulation.  
**Cons**: Cannot use intermediate predictions as features; fixed horizon size.

### 2. Recursive (Iterated) Strategy

Train a one-step-ahead model, then use its prediction as input for the next step:

```
ŷ_{t+1} = f(x_t, x_{t-1}, ...)
ŷ_{t+2} = f(ŷ_{t+1}, x_t, ...)
```

**Pros**: Single model, any horizon. Naturally matches the seq2seq decoder.  
**Cons**: Errors compound — a poor step-1 prediction degrades all subsequent predictions. Forecast variance grows with horizon.

### 3. Direct-Recursive Hybrid (Multi-Stage)

Train both a direct multi-step model and a recursive model, then combine them:

- Use the direct model's outputs as features for the recursive model.
- Or average predictions from both.
- Or use the direct prediction as a "correction" to the recursive path.

**Pros**: Reduces bias of direct models and variance of recursive models.  
**Cons**: More complex training pipeline; two models to maintain.

## Practical Considerations

- **Hidden size**: Start with 32–128 for univariate, 64–256 for multivariate. Tune via validation loss.
- **Number of layers**: 1–3 layers. More layers capture hierarchical temporal patterns but risk overfitting.
- **Dropout**: Add `nn.Dropout(0.2–0.5)` between LSTM layers to regularize.
- **Bidirectional LSTM**: Processes the sequence in both directions. Use only for *classification* or *imputation* tasks. For forecasting, bidirectional LSTMs leak future information and should be avoided.
- **Gradient clipping**: Clip gradients to a max norm (e.g., 1.0) to prevent exploding gradients.

## Summary

| Strategy | Error Propagation | Horizon Flexibility | When to Use |
|---|---|---|---|
| Direct | None | Fixed | Short, fixed horizon |
| Recursive | Compounds | Any | Variable horizon, long sequences |
| Hybrid | Reduced | Any | Best accuracy, more complexity |

LSTMs remain a strong baseline for time series forecasting. They excel with medium-length sequences (50–500 steps) and capture complex temporal dynamics that simpler models miss. For very long sequences, attention mechanisms or Transformers often outperform pure recurrent models.
