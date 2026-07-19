# Long Short-Term Memory (LSTM)

## Overview

The Long Short-Term Memory network, introduced by Hochreiter and Schmidhuber (1997), is a gated RNN architecture designed to overcome the vanishing gradient problem. LSTMs introduce a **cell state** $\mathbf{c}_t$ that acts as a memory highway, allowing gradients to flow through time with minimal attenuation.

## Core Idea: The Cell State

The cell state $\mathbf{c}_t$ runs straight through the entire sequence with only linear interactions (element-wise multiplication and addition), making gradient flow stable:

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$

The hidden state $\mathbf{h}_t$ is a filtered version of the cell state:

$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
$$

## The Three Gates

LSTMs use three gates to control information flow. Each gate is a sigmoid layer outputting values in $(0, 1)$, determining how much information passes through.

### Forget Gate $\mathbf{f}_t$

Determines what to discard from the previous cell state:

$$
\mathbf{f}_t = \sigma(\mathbf{W}_{hf} \mathbf{h}_{t-1} + \mathbf{W}_{xf} \mathbf{x}_t + \mathbf{b}_f)
$$

Each element of $\mathbf{f}_t$ is between $0$ (completely forget) and $1$ (completely retain).

### Input Gate $\mathbf{i}_t$

Controls what new information to add to the cell state:

$$
\mathbf{i}_t = \sigma(\mathbf{W}_{hi} \mathbf{h}_{t-1} + \mathbf{W}_{xi} \mathbf{x}_t + \mathbf{b}_i)
$$

### Candidate Cell State $\tilde{\mathbf{c}}_t$

The new candidate values (modulated by $\tanh$) that could be added:

$$
\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_{hc} \mathbf{h}_{t-1} + \mathbf{W}_{xc} \mathbf{x}_t + \mathbf{b}_c)
$$

### Output Gate $\mathbf{o}_t$

Determines what parts of the cell state to output:

$$
\mathbf{o}_t = \sigma(\mathbf{W}_{ho} \mathbf{h}_{t-1} + \mathbf{W}_{xo} \mathbf{x}_t + \mathbf{b}_o)
$$

## Complete LSTM Forward Pass

At each time step $t$, the LSTM computes:

1. **Forget gate**: $\mathbf{f}_t = \sigma(\mathbf{W}_{hf} \mathbf{h}_{t-1} + \mathbf{W}_{xf} \mathbf{x}_t + \mathbf{b}_f)$
2. **Input gate**: $\mathbf{i}_t = \sigma(\mathbf{W}_{hi} \mathbf{h}_{t-1} + \mathbf{W}_{xi} \mathbf{x}_t + \mathbf{b}_i)$
3. **Candidate cell**: $\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_{hc} \mathbf{h}_{t-1} + \mathbf{W}_{xc} \mathbf{x}_t + \mathbf{b}_c)$
4. **Cell state update**: $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$
5. **Output gate**: $\mathbf{o}_t = \sigma(\mathbf{W}_{ho} \mathbf{h}_{t-1} + \mathbf{W}_{xo} \mathbf{x}_t + \mathbf{b}_o)$
6. **Hidden state**: $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$

Where $\sigma$ is the sigmoid function: $\sigma(x) = \frac{1}{1 + e^{-x}}$, and $\odot$ denotes element-wise (Hadamard) product.

## Why LSTM Solves the Vanishing Gradient Problem

### Constant Error Carousel (CEC)

The key innovation is the additive update to the cell state:

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$

Consider the gradient of $\mathcal{L}$ with respect to $\mathbf{c}_{t-1}$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{c}_{t-1}} = \frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} \frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} \cdot \mathbf{f}_t
$$

If the forget gate $\mathbf{f}_t$ is close to $1$ (which the network can learn), the gradient flows through unchanged:

$$
\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} \approx 1 \quad \Rightarrow \quad \frac{\partial \mathcal{L}}{\partial \mathbf{c}_{t-1}} \approx \frac{\partial \mathcal{L}}{\partial \mathbf{c}_t}
$$

This means the error can propagate backward through many time steps without vanishing or exploding. The additive interaction replaces the multiplicative Jacobian chain of vanilla RNNs with a near-identity transformation.

### Comparison with Vanilla RNNs

| Aspect | Vanilla RNN | LSTM |
|--------|-------------|------|
| State update | $\mathbf{h}_t = \tanh(\mathbf{W}[\mathbf{h}_{t-1}; \mathbf{x}_t])$ | $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$ |
| Gradient flow | Product of many $\tanh' \cdot \mathbf{W}$ terms | Linear through $\mathbf{f}_t$ (learnable near 1) |
| Long-range dependencies | Poor beyond ~10 steps | Can handle 100+ steps |
| Parameters | $O(d_h^2 + d_h d_x)$ | $4 \times O(d_h^2 + d_h d_x)$ |

### When the Forget Gate is Open

When $\mathbf{f}_t \approx \mathbf{1}$ and $\mathbf{i}_t \approx \mathbf{0}$, the cell state simply persists: $\mathbf{c}_t \approx \mathbf{c}_{t-1}$. The network learns to open and close the forget gate, creating clean memory highways. This is why LSTMs can remember information over thousands of time steps.

## Peephole Connections

An extension proposed by Gers and Schmidhuber (2000) adds "peephole" connections that let the gates see the cell state:

$$
\mathbf{f}_t = \sigma(\mathbf{W}_{hf} \mathbf{h}_{t-1} + \mathbf{W}_{xf} \mathbf{x}_t + \mathbf{W}_{cf} \mathbf{c}_{t-1} + \mathbf{b}_f)
$$

$$
\mathbf{i}_t = \sigma(\mathbf{W}_{hi} \mathbf{h}_{t-1} + \mathbf{W}_{xi} \mathbf{x}_t + \mathbf{W}_{ci} \mathbf{c}_{t-1} + \mathbf{b}_i)
$$

$$
\mathbf{o}_t = \sigma(\mathbf{W}_{ho} \mathbf{h}_{t-1} + \mathbf{W}_{xo} \mathbf{x}_t + \mathbf{W}_{co} \mathbf{c}_t + \mathbf{b}_o)
$$

Peephole connections allow the gates to directly inspect the cell state, enabling more precise timing and counting. In practice, their benefit is modest and many modern implementations omit them.

## Variants and Practical Considerations

- **Bidirectional LSTM**: Processes sequences in both forward and backward directions.
- **Stacked LSTM**: Multiple LSTM layers on top of each other for hierarchical feature learning.
- **Dropout**: Applied between LSTM layers (not recurrent connections) to prevent overfitting.
- **Gradient clipping**: Still needed to prevent exploding gradients during training.

## Practical Usage Notes

When training LSTMs, it is standard to initialize the forget gate bias to $1$ or a large positive value (e.g., $b_f = 1$) so the forget gate is initially open, preserving gradient flow early in training. Gradient clipping with a threshold of $5$ or $10$ is almost always necessary to prevent the rare but destructive exploding gradients that still occur despite the LSTM's improved gradient flow.

## Summary

LSTMs revolutionized sequence modeling by introducing a gated cell state that enables stable gradient flow over long sequences. The forget gate is the critical component that allows the network to learn when to reset its memory and when to preserve information indefinitely.
