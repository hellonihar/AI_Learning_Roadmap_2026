# Bidirectional and Stacked RNNs

## Bidirectional RNNs

A standard RNN processes a sequence from left to right, so each hidden state $\overrightarrow{\mathbf{h}}_t$ only has information about the past (inputs $\mathbf{x}_1$ through $\mathbf{x}_t$). A **Bidirectional RNN (BiRNN)** processes the sequence in both directions, giving each time step access to past and future context.

### Architecture

A BiRNN maintains two separate hidden states:

- **Forward RNN**: $\overrightarrow{\mathbf{h}}_t = \text{RNN}_{\text{fwd}}(\mathbf{x}_t, \overrightarrow{\mathbf{h}}_{t-1})$
- **Backward RNN**: $\overleftarrow{\mathbf{h}}_t = \text{RNN}_{\text{bwd}}(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1})$

The final representation at time $t$ is formed by concatenating (or summing) both hidden states:

$$
\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t ; \overleftarrow{\mathbf{h}}_t]
$$

The forward RNN reads the input normally; the backward RNN reads the input in **reverse order** (from $\mathbf{x}_T$ to $\mathbf{x}_1$). The two RNNs have **no interaction** — they are independent networks with separate parameters.

### Applications

- **Named Entity Recognition**: determining whether a word is an entity name benefits from context on both sides.
- **Part-of-Speech Tagging**: the tag of a word depends on surrounding words in both directions.
- **Machine Translation**: encoding a source sentence requires understanding the full context.

### Limitations

- **Not causal**: the representation at time $t$ depends on future inputs, making BiRNNs unsuitable for real-time or streaming applications.
- **Higher latency**: the full sequence must be available before any output can be produced.
- **More parameters**: roughly double the parameters of a unidirectional RNN.

## Stacked (Multi-Layer) RNNs

Stacked RNNs (also called deep RNNs) consist of multiple RNN layers on top of each other. The hidden state of layer $l$ at time $t$ becomes the input to layer $l+1$ at the same time step.

### Architecture

For a stack of $L$ layers:

$$
\mathbf{h}_t^{(1)} = \text{RNN}^{(1)}(\mathbf{x}_t, \mathbf{h}_{t-1}^{(1)})
$$

$$
\mathbf{h}_t^{(l)} = \text{RNN}^{(l)}(\mathbf{h}_t^{(l-1)}, \mathbf{h}_{t-1}^{(l)}) \quad \text{for } l = 2, \dots, L
$$

Each layer operates at a different level of abstraction:
- **Lower layers**: capture local, short-term patterns (e.g., character-level features in text, raw signal patterns in audio).
- **Higher layers**: capture global, long-term structure (e.g., sentence-level semantics, high-level motifs in music).

### Why Stack RNNs?

1. **Hierarchical representation learning**: deeper layers model more abstract features.
2. **Increased capacity**: more parameters and nonlinearities allow modeling complex functions.
3. **Empirical gains**: stacked LSTMs consistently outperform single-layer ones on language modeling, machine translation, and speech recognition.

### Typical Depth

| Task | Typical Depth |
|------|--------------|
| Character-level language modeling | 2–3 layers |
| Word-level language modeling | 2–4 layers |
| Machine translation (encoder) | 4–8 layers |
| Speech recognition | 5–7 layers |

## Training Considerations for Deep RNNs

### Gradient Propagation

Gradients must flow through both time (BPTT) and depth (layer-to-layer). This compounds the vanishing/exploding gradient problem. Skip connections or residual connections between layers can help.

### Computational Cost

Training deep RNNs is expensive — both time and memory scale as $O(L \cdot T \cdot d_h^2)$. Unrolling all time steps for all layers requires storing all intermediate activations.

### Regularization

- **Dropout**: Applied between layers (not on recurrent connections) to prevent co-adaptation.
- **Zoneout**: Stochastic variant where some hidden units are randomly preserved across time.
- **Variational dropout**: Same dropout mask applied across all time steps.

### Initialization

- **Layer-wise learning rates**: lower layers may need different learning rates than higher layers.
- **Gradient clipping**: essential to prevent exploding gradients in deep unrolled graphs.

## Summary

Bidirectional RNNs provide full sequence context by processing in both directions, at the cost of causality. Stacked RNNs build hierarchical representations through multiple layers, increasing capacity at the cost of training complexity. Combining both (stacked bidirectional RNNs) is a common pattern for sequence encoding tasks.
