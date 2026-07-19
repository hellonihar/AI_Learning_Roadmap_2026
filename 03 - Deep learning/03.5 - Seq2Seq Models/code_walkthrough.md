# Code Walkthrough: Attention Mechanisms with NumPy

This walkthrough implements Bahdanau attention scoring, scaled dot-product attention, and a minimal Seq2Seq model for a synthetic sequence reversal task using **only NumPy**. No deep learning frameworks are used — the goal is to understand the mechanics.

## Setup

```python
import numpy as np
```

## 1. Bahdanau (Additive) Attention Scoring

Given a decoder hidden state $s_{t-1}$ and encoder hidden states $h_1, \dots, h_T$, compute alignment scores:

$$e_{t,i} = v_a^\top \tanh(W_a s_{t-1} + U_a h_i)$$

```python
def bahdanau_score(decoder_hidden, encoder_states, W_a, U_a, v_a):
    """
    decoder_hidden:  (d_hidden,)
    encoder_states:  (T, d_hidden)
    W_a:             (d_attn, d_hidden)
    U_a:             (d_attn, d_hidden)
    v_a:             (d_attn,)
    Returns:         (T,) alignment scores
    """
    # Project decoder hidden: (d_attn,)
    proj_dec = W_a @ decoder_hidden
    # Project encoder states: (T, d_attn)
    proj_enc = encoder_states @ U_a.T
    # Combine: broadcast proj_dec to (T, d_attn), then tanh
    scores = np.tanh(proj_enc + proj_dec)  # (T, d_attn)
    # Compute v_a^T * tanh(...) -> (T,)
    scores = scores @ v_a
    return scores
```

### Attention Weights and Context Vector

```python
def compute_attention(scores):
    """Convert scores to weights and compute context vector."""
    # Softmax over encoder dimension
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)  # (T,)
    return weights
```

## 2. Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, seq_len_q, d_k)
    K: (batch, seq_len_k, d_k)
    V: (batch, seq_len_k, d_v)
    mask: (batch, seq_len_q, seq_len_k) or None
    Returns: (batch, seq_len_q, d_v), (batch, seq_len_q, seq_len_k) weights
    """
    d_k = K.shape[-1]
    # scores: (batch, seq_len_q, seq_len_k)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    # Softmax over keys (last dim)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Weighted sum of values: (batch, seq_len_q, d_v)
    output = weights @ V
    return output, weights
```

### Multi-Head Attention (Conceptual)

Multi-head runs $h$ parallel attention operations:

```python
def multi_head_attention(Q, K, V, h, W_q, W_k, W_v, W_o):
    """Conceptual multi-head attention (single batch item)."""
    d_model = Q.shape[-1]
    d_k = d_v = d_model // h

    def split_heads(x):
        """Split last dim into (h, d_k), transpose to (h, seq, d_k)."""
        x = x.reshape(x.shape[0], h, d_k)
        return x.transpose(1, 0, 2)

    # Linear projections: (seq, d_model) -> (seq, d_model)
    Q_p = Q @ W_q  # (seq, d_model)
    K_p = K @ W_k
    V_p = V @ W_v

    # Split into h heads: (h, seq, d_k)
    Q_heads = split_heads(Q_p)  # shape (h, seq_q, d_k)
    K_heads = split_heads(K_p)
    V_heads = split_heads(V_p)

    # Apply attention per head
    outputs = []
    for i in range(h):
        out, w = scaled_dot_product_attention(
            Q_heads[i], K_heads[i], V_heads[i]
        )
        outputs.append(out)  # each: (seq_q, d_k)

    # Concatenate heads: (seq_q, d_model)
    concat = np.concatenate(outputs, axis=-1)
    return concat @ W_o
```

## 3. Simple Encoder-Decoder with Attention

We implement a minimal Seq2Seq for **sequence reversal** (e.g., `[1, 2, 3]` -> `[3, 2, 1]`). We use simple RNN cells (identity activation) for clarity.

```python
class SimpleRNNCell:
    def __init__(self, input_size, hidden_size):
        self.W_h = np.random.randn(hidden_size, input_size) * 0.01
        self.U_h = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros(hidden_size)

    def forward(self, x, h_prev):
        # h_t = tanh(W_h x_t + U_h h_{t-1} + b_h)
        h = np.tanh(self.W_h @ x + self.U_h @ h_prev + self.b_h)
        return h


class Seq2SeqWithAttention:
    def __init__(self, vocab_size, hidden_size):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_dim = hidden_size

        # Embeddings
        self.embed = np.random.randn(vocab_size, self.embed_dim) * 0.1

        # RNN cells
        self.enc_cell = SimpleRNNCell(self.embed_dim, hidden_size)
        self.dec_cell = SimpleRNNCell(self.embed_dim, hidden_size)

        # Attention parameters (Bahdanau)
        self.W_a = np.random.randn(hidden_size, hidden_size) * 0.01
        self.U_a = np.random.randn(hidden_size, hidden_size) * 0.01
        self.v_a = np.random.randn(hidden_size) * 0.01

        # Output projection
        self.W_o = np.random.randn(vocab_size, hidden_size * 2) * 0.01
        self.b_o = np.zeros(vocab_size)

    def encode(self, input_ids):
        T = len(input_ids)
        h = np.zeros(self.hidden_size)
        states = []
        for t in range(T):
            x = self.embed[input_ids[t]]
            h = self.enc_cell.forward(x, h)
            states.append(h.copy())
        return states

    def decode_step(self, y_embed, prev_h, encoder_states):
        # Compute attention over encoder states
        scores = bahdanau_score(prev_h, encoder_states,
                                self.W_a, self.U_a, self.v_a)
        attn_weights = compute_attention(scores)
        context = np.sum(encoder_states * attn_weights[:, None], axis=0)

        # RNN step
        h = self.dec_cell.forward(y_embed, prev_h)

        # Concatenate decoder state and context
        combined = np.concatenate([h, context])

        # Output projection -> vocabulary logits
        logits = self.W_o @ combined + self.b_o
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        return h, context, probs, attn_weights

    def forward(self, input_ids, target_ids=None):
        """For training: returns loss. For inference: returns predictions."""
        encoder_states = self.encode(input_ids)
        T_out = len(target_ids) if target_ids is not None else len(input_ids)

        # Start with <SOS> token (assumed index 0)
        dec_h = np.zeros(self.hidden_size)
        y_embed = self.embed[0]

        logits_list = []
        self.last_attention_weights = []

        for t in range(T_out):
            dec_h, context, probs, attn_w = self.decode_step(
                y_embed, dec_h, encoder_states)
            logits_list.append(np.log(probs + 1e-8))
            self.last_attention_weights.append(attn_w)

            if target_ids is not None:
                # Teacher forcing
                y_embed = self.embed[target_ids[t]]
            else:
                # Autoregressive (greedy)
                pred = np.argmax(probs)
                y_embed = self.embed[pred]

        return np.array(logits_list)
```

### Training Loop (Conceptual)

```python
def train_epoch(model, data, lr=0.001):
    total_loss = 0
    for input_ids, target_ids in data:
        # Forward
        log_probs = model.forward(input_ids, target_ids)
        # Cross-entropy loss
        losses = [-log_probs[t, target_ids[t]]
                  for t in range(len(target_ids))]
        loss = sum(losses) / len(target_ids)
        # (Gradient descent omitted for brevity; would compute and
        #  update all parameters via backprop through time)
        total_loss += loss
    return total_loss / len(data)
```

## 4. Attention Weight Matrix Visualization

After training on the reversal task, we can inspect the attention weights:

```python
def visualize_attention(model, input_ids):
    """Run inference and display attention weights as a matrix."""
    _ = model.forward(input_ids, target_ids=input_ids)
    attn_matrix = np.array(model.last_attention_weights)  # (T_dec, T_enc)

    print("Input: ", input_ids)
    print("Attention weights (rows=decoder steps, cols=encoder steps):")
    for row in attn_matrix:
        print(" ".join(f"{v:.2f}" for v in row))

    # For a reversal task, a diagonal-ish pattern indicates
    # the model has learned to align each output position
    # to the corresponding input position in reverse order.
```

### Expected Visualization (Reversal Task)

For input `[1, 2, 3]` -> target `[3, 2, 1]`:

```
         Encoder positions
         h_1   h_2   h_3
Dec t=1  0.02  0.08  0.90   <- attending to input[3] (value 3)
Dec t=2  0.05  0.88  0.07   <- attending to input[2] (value 2)
Dec t=3  0.91  0.07  0.02   <- attending to input[1] (value 1)
```

The diagonal pattern (anti-diagonal for reversal) confirms that attention correctly aligns source and target positions.

## Summary

This walkthrough shows the core mechanics of attention:

- **Bahdanau scoring**: A small neural network computes alignment between decoder state and each encoder state.
- **Scaled dot-product**: Simple dot products scaled by $\sqrt{d_k}$ — the foundation of the Transformer.
- **Seq2Seq with attention**: The decoder builds a dynamic context vector at each step by attending to relevant encoder states.
- **Visualization**: Attention weights provide interpretability — we can see what the model "looks at" during generation.
