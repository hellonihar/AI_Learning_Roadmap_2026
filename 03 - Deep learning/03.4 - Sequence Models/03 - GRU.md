# Gated Recurrent Unit (GRU)

## Overview

The Gated Recurrent Unit, introduced by Cho et al. (2014), is a simplified gated RNN architecture that combines the forget and input gates of the LSTM into a single **update gate**. GRUs have fewer parameters than LSTMs while achieving comparable performance on many tasks.

## GRU Architecture

A GRU has two gates: a **reset gate** and an **update gate**. Unlike the LSTM, the GRU does not maintain a separate cell state — it uses a single hidden state $\mathbf{h}_t$ that serves both as memory and output.

### Reset Gate $\mathbf{r}_t$

Controls how much of the previous hidden state to forget when computing the candidate state:

$$
\mathbf{r}_t = \sigma(\mathbf{W}_{hr} \mathbf{h}_{t-1} + \mathbf{W}_{xr} \mathbf{x}_t + \mathbf{b}_r)
$$

When $\mathbf{r}_t \approx \mathbf{0}$, the network ignores the past and resets its memory.

### Update Gate $\mathbf{z}_t$

Controls how much of the previous hidden state to carry forward versus how much to update with new information:

$$
\mathbf{z}_t = \sigma(\mathbf{W}_{hz} \mathbf{h}_{t-1} + \mathbf{W}_{xz} \mathbf{x}_t + \mathbf{b}_z)
$$

This is analogous to the LSTM's forget and input gates combined into one.

### Candidate Hidden State $\tilde{\mathbf{h}}_t$

The new candidate memory, modulated by the reset gate:

$$
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_{hh} (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)
$$

The reset gate determines whether the candidate state depends on the previous hidden state or is computed fresh from the input.

### Final Hidden State

The update gate interpolates between the old hidden state and the candidate:

$$
\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t
$$

When $\mathbf{z}_t \approx \mathbf{1}$, the network mostly retains the old state (like LSTM's forget gate ≈ 1). When $\mathbf{z}_t \approx \mathbf{0}$, the network mostly adopts the new candidate.

## Complete GRU Forward Pass

1. $\mathbf{r}_t = \sigma(\mathbf{W}_{hr} \mathbf{h}_{t-1} + \mathbf{W}_{xr} \mathbf{x}_t + \mathbf{b}_r)$
2. $\mathbf{z}_t = \sigma(\mathbf{W}_{hz} \mathbf{h}_{t-1} + \mathbf{W}_{xz} \mathbf{x}_t + \mathbf{b}_z)$
3. $\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_{hh} (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$
4. $\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t$

## GRU vs LSTM: Comparison

| Aspect | LSTM | GRU |
|--------|------|-----|
| **Gates** | 3 (forget, input, output) | 2 (reset, update) |
| **States** | Cell state $\mathbf{c}_t$ + Hidden state $\mathbf{h}_t$ | Hidden state $\mathbf{h}_t$ only |
| **Parameter count** | $4(d_h^2 + d_h d_x + d_h)$ | $3(d_h^2 + d_h d_x + d_h)$ |
| **Gradient flow** | Via $\mathbf{c}_t$ with forget gate | Via $\mathbf{h}_t$ with update gate |
| **Expressiveness** | Separate memory and output | Unified memory-output |
| **Typical performance** | Comparable or slightly better on complex tasks | Comparable, often faster to train |

### Parameter Comparison

For hidden size $d_h$ and input size $d_x$:

- LSTM parameters per cell: $4(d_h d_x + d_h^2 + d_h)$
- GRU parameters per cell: $3(d_h d_x + d_h^2 + d_h)$

GRU uses roughly **25% fewer parameters** than an equivalently-sized LSTM.

## When to Use Which

### Prefer GRU when:
- Dataset is small (fewer parameters reduce overfitting)
- Computational budget is limited (faster training, less memory)
- Task does not require extremely long-range dependencies
- You want a simpler, easier-to-tune architecture

### Prefer LSTM when:
- Dataset is large and complex
- Task requires capturing very long-range dependencies (100+ steps)
- You need separate control over memory retention and output filtering
- Strong empirical baseline is needed (LSTM is more battle-tested)

### Practical Notes

- For most NLP tasks (translation, sentiment analysis), both perform similarly with proper tuning.
- GRU often converges faster but LSTM may achieve slightly better final performance on complex tasks.
- Modern practice increasingly uses Transformers, but RNNs (especially LSTMs) remain strong for time series, streaming data, and resource-constrained settings.
- Bidirectional variants of both architectures consistently outperform unidirectional ones.

## Summary

GRU simplifies the LSTM by merging the forget and input gates into an update gate and removing the separate cell state. This reduces parameters and computational cost while maintaining competitive performance. The choice between GRU and LSTM depends on the specific task, dataset size, and computational constraints.
