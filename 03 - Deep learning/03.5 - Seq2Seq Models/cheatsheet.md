# Cheatsheet: Seq2Seq Models & Attention

## Seq2Seq (Encoder-Decoder) Flow

```
Input Sequence (x₁, x₂, ..., x_T)
        │
        ▼
    ┌──────────────┐
    │   Encoder     │  RNN/LSTM reads each token
    │  (RNN/LSTM)   │  h_t = f(x_t, h_{t-1})
    └──────┬───────┘
           │
     Context Vector c = h_T (final hidden state)
           │
           ▼
    ┌──────────────┐
    │   Decoder     │  Generates y_t conditioned on c and past outputs
    │  (RNN/LSTM)   │  s_t = g(y_{t-1}, s_{t-1}, c)
    └──────┬───────┘           P(y_t | ...) = softmax(W_o s_t + b_o)
           │
           ▼
Output Sequence (y₁, y₂, ..., y_T')
```

**Teacher Forcing**: Train decoder with ground-truth previous token instead of own prediction. Mitigates exposure bias via scheduled sampling.

---

## Attention Mechanisms

### Bahdanau (Additive) Attention

| Step | Formula |
|---|---|
| Alignment score | $e_{t,i} = v_a^\top \tanh(W_a s_{t-1} + U_a h_i)$ |
| Attention weights | $\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{k=1}^{T} \exp(e_{t,k})}$ |
| Context vector | $c_t = \sum_{i=1}^{T} \alpha_{t,i} h_i$ |

### Luong Attention Variants

| Variant | Score Formula $e_{t,i}$ |
|---|---|
| Dot | $s_t^\top h_i$ |
| General | $s_t^\top W_a h_i$ |
| Concat | $v_a^\top \tanh(W_a [s_t; h_i])$ |

### Comparison

| Aspect | Bahdanau | Luong Dot | Luong General |
|---|---|---|---|
| Decoder state | $s_{t-1}$ | $s_t$ | $s_t$ |
| Parameters | $W_a, U_a, v_a$ | None | $W_a$ |
| Complexity per pair | $O(d_h^2)$ | $O(d_h)$ | $O(d_h^2)$ |

---

## Transformer Architecture

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

### Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

### Transformer Components

| Component | Description |
|---|---|
| Self-attention | Each position attends to all positions in same sequence |
| Cross-attention | Decoder attends to encoder outputs (Q from decoder, K,V from encoder) |
| Position-wise FFN | $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$ |
| Residual connection | $\text{output} = x + \text{SubLayer}(x)$ |
| Layer normalization | $\text{LayerNorm}(x) = \gamma \odot \frac{x-\mu}{\sigma} + \beta$ |
| Positional encoding | $PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$, $PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d})$ |

### Architecture Stack

```
Encoder (×N):  [Self-Attention → FFN]  (with residual + layer norm)
Decoder (×N):  [Masked Self-Attention → Cross-Attention → FFN]  (with residual + layer norm)
```

---

## Masking Types

| Mask | Purpose | Shape | Applied to |
|---|---|---|---|
| Padding mask | Ignore `<PAD>` tokens | `[1, 1, 1, seq_len]` | Encoder & decoder self-attn, cross-attn |
| Look-ahead mask | Prevent attending to future | `[seq_len, seq_len]` | Decoder self-attention only |
| Combined | Padding + causal | OR of both | Decoder self-attention |

Applied as: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$ (masked entries = $-\infty$)

---

## BERT vs. GPT

| Property | BERT | GPT |
|---|---|---|
| Transformer part | Encoder-only | Decoder-only |
| Attention | Bidirectional | Causal (left-to-right) |
| Pre-training | MLM + NSP | Autoregressive LM |
| Context | Full (both directions) | Unidirectional (left) |
| Generation | Not supported | Autoregressive |
| Fine-tuning | Task-specific heads | Prompt-based or fine-tune |
| Typical uses | Classification, NER, QA | Generation, chatbots, code |

---

## Key Implementation Dimensions (Transformer Base)

| Hyperparameter | Value |
|---|---|
| $d_{\text{model}}$ | 512 |
| $h$ (heads) | 8 |
| $d_k = d_v$ | 64 |
| $d_{ff}$ | 2048 |
| $N$ (layers) | 6 |
\end{document}
