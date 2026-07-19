# Positional Encoding

## Why Position Matters

RNNs process tokens sequentially, with position inherently encoded in the order of recurrence. The Transformer processes all tokens in parallel through self-attention — which is **permutation-invariant**: swapping two tokens produces the same output (just reordered). Without positional information, the model would see "I ate the cake" and "the cake ate I" as identical sequences.

To fix this, positional encodings are **added** to the input embeddings before the first encoder/decoder layer.

## Sinusoidal Positional Encoding

The original Transformer uses fixed sinusoidal functions. For position $pos$ and dimension $i$:

$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\end{aligned}
$$

where $d_{\text{model}}$ is the embedding dimension (e.g., 512), and $i$ ranges from $0$ to $d_{\text{model}}/2 - 1$.

### Properties

1. **Each position has a unique encoding** — the combination of sines and cosines at different frequencies produces a distinct vector for each position.
2. **Relative position information**: The encoding at position $pos+k$ can be represented as a linear function of the encoding at position $pos$, because:
   $$\sin(pos + k) = \sin(pos)\cos(k) + \cos(pos)\sin(k)$$
   This allows the model to easily learn relative position dependencies.
3. **Different frequencies**: Lower dimensions use lower frequencies (longer periods), encoding coarse position information; higher dimensions use higher frequencies, encoding fine-grained position distinctions.
4. **Bounded values**: Outputs are in $[-1, 1]$, making them compatible with learned embeddings.
5. **No learned parameters**: The encoding is deterministic and does not require training, making it robust to sequences longer than those seen during training.

The frequency term $\omega_i = 1/10000^{2i/d_{\text{model}}}$ creates a geometric progression from $2\pi \cdot 10000^{0} \approx 2\pi$ to $2\pi \cdot 10000^{-1} \approx 2\pi \cdot 0.0001$.

## Learned Positional Encoding

An alternative is to treat positional encodings as learnable parameters, same as token embeddings. The model learns a matrix $P \in \mathbb{R}^{L_{\max} \times d_{\text{model}}}$ where row $pos$ is the encoding for position $pos$.

- **Pros**: Can adapt to task-specific position patterns; simpler conceptually.
- **Cons**: Cannot handle sequences longer than $L_{\max}$ seen during training; requires more parameters.

BERT and GPT use learned positional embeddings.

## Relative Position Representations

Shaw et al. (2018) proposed encoding relative distances between tokens rather than absolute positions. Instead of learning $PE(pos)$, the model learns $a_{ij}^K$ and $a_{ij}^V$ for the relative distance $j - i$ between positions $i$ and $j$, clipped to a maximum offset $k$:

$$a_{ij}^K = w_{\text{clip}(j-i, -k, k)}^K$$

These are incorporated into the attention computation:

$$e_{ij} = \frac{x_i W^Q (x_j W^K + a_{ij}^K)^\top}{\sqrt{d_k}}$$

Relative representations generalize better to longer sequences than absolute encodings and have been adopted in models like Transformer-XL and T5.

## Summary

| Approach | Type | Extrapolation | Parameters |
|---|---|---|---|
| Sinusoidal | Fixed | Yes (theoretically) | None |
| Learned | Learned | No | $L_{\max} \cdot d_{\text{model}}$ |
| Relative | Learned | Partial (clipping) | $(2k+1) \cdot d_k$ |

Positional encoding is a small but critical component — it is the only source of order information in the Transformer, and its design directly impacts the model's ability to generalize to sequence length and structure.
