# Masking Strategies

## Why Masks Are Needed

In the Transformer and Seq2Seq models, two fundamental problems require masking:

1. **Variable-length sequences**: Batches contain sequences of different lengths. Padding tokens must be ignored in attention computations.
2. **Autoregressive generation**: The decoder must not attend to future tokens, which do not exist yet during inference and must be hidden during training.

## Padding Mask

When batching variable-length sequences, shorter sequences are padded with a special `<PAD>` token to the length of the longest sequence. The padding mask prevents the model from attending to these meaningless tokens.

In attention, the mask is applied before softmax:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$$

where $M$ is a mask matrix with entries:

$$M_{ij} = \begin{cases}
0 & \text{if position } j \text{ is valid (not padding)} \\
-\infty & \text{if position } j \text{ is padding}
\end{cases}$$

Adding $-\infty$ ensures that after softmax, the attention weight for the padding position becomes zero. This mask is used in:

- **Encoder self-attention**: Prevents attending to padding positions in the input.
- **Decoder cross-attention** (encoder-decoder attention): Same — the decoder should not attend to padding in the encoder output.
- **Decoder self-attention**: Prevents attending to padding in the target sequence.

### Implementation

A padding mask is typically a boolean tensor of shape `[batch_size, 1, 1, seq_len]` (broadcastable to the attention score shape `[batch_size, num_heads, query_len, key_len]`), where `True` indicates a position to mask.

## Look-Ahead (Causal) Mask

During training, the decoder receives the **entire target sequence** at once. Without masking, each position could attend to **future positions**, leaking information that would not be available during inference.

The look-ahead mask (also called causal mask or autoregressive mask) prevents position $i$ from attending to position $j$ where $j > i$:

$$M_{ij} = \begin{cases}
0 & \text{if } j \leq i \text{ (allowed to attend)} \\
-\infty & \text{if } j > i \text{ (future, masked)}
\end{cases}$$

This creates a lower-triangular attention pattern — each token can only attend to itself and preceding tokens.

### Implementation

The look-ahead mask is a boolean matrix of shape `[seq_len, seq_len]` with `True` in the upper triangle (positions to mask). It is applied in the decoder's **self-attention** sub-layer during training.

## Combined Mask in Decoder

In practice, the decoder self-attention uses the **union** of both masks:

$$M_{\text{combined}} = M_{\text{look-ahead}} \text{ OR } M_{\text{padding}}$$

A position is masked if it is either a future position or a padding token. This ensures the decoder respects both the causal structure of generation and the variable-length nature of the target sequence.

## Masking in Different Architectures

| Architecture | Padding Mask | Look-Ahead Mask |
|---|---|---|
| Encoder (Transformer) | Yes | No |
| Decoder (Transformer) | Yes | Yes |
| BERT (Encoder-only) | Yes | No |
| GPT (Decoder-only) | Yes | Yes |

## Summary

- **Padding mask**: Ignores `<PAD>` tokens due to variable-length sequences.
- **Look-ahead mask**: Prevents attending to future positions, preserving autoregressive property.
- **Combined**: Both masks are applied together in the decoder self-attention.
- **Implementation**: Masks are added to attention scores before softmax, with masked positions set to $-\infty$.

Without these masks, the Transformer would either attend to meaningless padding (hurting quality) or cheat by looking at future tokens (making training unrealistic).
