# Transformer Architecture

## Introduction

The Transformer, introduced by Vaswani et al. (2017) in "Attention Is All You Need", completely abandons recurrence and convolution, relying entirely on attention mechanisms to model dependencies between sequence elements. It achieves parallel computation, superior long-range dependency modeling, and state-of-the-art results on machine translation and virtually every subsequent NLP task.

## High-Level Structure

The Transformer follows the encoder-decoder paradigm but with a fundamentally different internal design. Both encoder and decoder are stacks of identical layers (typically $N=6$).

### Encoder Stack

Each encoder layer has two sub-layers:

1. **Multi-head self-attention** — allows each position to attend to all positions in the input.
2. **Position-wise feed-forward network (FFN)** — a two-layer MLP applied independently at each position.

Each sub-layer is wrapped with a **residual connection** followed by **layer normalization**:

$$\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

### Decoder Stack

Each decoder layer has three sub-layers:

1. **Masked multi-head self-attention** — self-attention over the decoder's own outputs, masked to prevent attending to future positions.
2. **Cross-attention** (encoder-decoder attention) — attends over the encoder's output; queries come from the decoder, keys and values from the encoder.
3. **Position-wise feed-forward network**.

The same residual + layer-norm wrapping applies to all sub-layers.

## Scaled Dot-Product Attention

The fundamental attention operation in the Transformer. Given queries $Q$, keys $K$, and values $V$ (all matrices), the output is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where $d_k$ is the dimension of the keys. The steps are:

1. **Compute scores**: $S = QK^\top$ produces an $(n \times m)$ matrix where entry $(i,j)$ is the dot product between query $i$ and key $j$.
2. **Scale**: Divide by $\sqrt{d_k}$ to prevent dot products from growing large in magnitude, which would push the softmax into regions with extremely small gradients.
3. **Mask** (optional): Apply a mask by setting certain entries to $-\infty$ before softmax.
4. **Softmax**: Normalize each row to produce attention weights (probability distribution over keys for each query).
5. **Weighted sum**: Multiply by $V$ to get the output for each query.

### Why Scaling Matters

Without scaling, for large $d_k$, the dot products grow large in magnitude, pushing the softmax into regions of extremely small gradients, which hurts training.

## Multi-Head Attention

Instead of performing a single attention operation, the Transformer projects the queries, keys, and values $h$ times with different learned linear projections, performs attention in parallel, concatenates the results, and projects again:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

where each head:

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

with projection matrices $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, and $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$.

In the original paper: $h = 8$, $d_k = d_v = d_{\text{model}} / h = 64$, $d_{\text{model}} = 512$.

Multi-head attention allows the model to attend to information from different representation subspaces at different positions, capturing diverse relationships (e.g., syntactic, semantic, positional). Empirically, different heads learn distinct roles — some focus on nearby tokens (local syntax), others on distant tokens (long-range semantics), and some on specific linguistic phenomena like dependency relations or coreference.

### Why Reduce Dimensions Per Head

Reducing $d_k$ and $d_v$ to $d_{\text{model}}/h$ keeps the total computational cost of multi-head attention equivalent to single-head attention with $d_{\text{model}}$-dimensional keys and values. Each head operates in a smaller subspace, but together they cover the full $d_{\text{model}}$ space, allowing the model to jointly attend to different representation aspects without increasing FLOPs.

## Input Embeddings and Output Projection

The Transformer uses learned embeddings to convert input and output tokens to $d_{\text{model}}$-dimensional vectors. The same weight matrix is typically shared between the input embedding layer and the pre-softmax output projection layer (weight tying), which reduces the total parameter count and acts as a regularizer. The embeddings are multiplied by $\sqrt{d_{\text{model}}}$ to match the scale of the positional encodings, ensuring that neither the semantic nor positional signal dominates at initialization.

## Positional Encoding

Since the Transformer processes all tokens simultaneously, it has no inherent notion of token order. Positional encodings are added to the input embeddings to inject positional information:

$$\text{input}_i = \text{embedding}(x_i) + \text{positional\_encoding}(i)$$

The original paper uses sinusoidal functions (see section "04 - Positional Encoding.md" for details).

## Position-Wise Feed-Forward Network

Each layer contains a fully connected feed-forward network applied identically to each position:

$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

This is a two-layer network with a ReLU activation in the hidden layer. The inner dimension $d_{ff}$ is typically much larger than $d_{\text{model}}$ (e.g., 2048 vs. 512).

## Residual Connections and Layer Normalization

Every sub-layer is followed by a residual connection and layer normalization:

$$\text{LayerNorm}(x + \text{Sublayer}(x))$$

**Residual connections** allow gradients to flow directly through the network, enabling training of deep models (the original has $6 \times 2 = 12$ sub-layers in the encoder and $6 \times 3 = 18$ in the decoder).

**Layer normalization** normalizes activations across the feature dimension (as opposed to batch normalization which normalizes across the batch):

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta$$

where $\mu$ and $\sigma$ are the mean and standard deviation computed over the feature dimension of $x$, and $\gamma$, $\beta$ are learned parameters.

## Encoder-Decoder Cross-Attention

In the decoder, each layer has a cross-attention sub-layer where:
- **Queries** come from the previous decoder layer
- **Keys** and **Values** come from the output of the encoder stack

This is how the decoder incorporates information from the input sequence. Each decoder position can attend to all encoder positions.

## Autoregressive Decoding and Masking

During training, the decoder receives the entire target sequence. To prevent it from "cheating" by looking ahead, **look-ahead masking** is applied in the self-attention sub-layer: position $i$ can only attend to positions $j \leq i$. This is implemented by setting attention scores for $j > i$ to $-\infty$ before softmax.

During inference, the decoder generates tokens one at a time, appending each predicted token to the input for the next step (autoregressive generation).

## Why the Transformer Works

1. **Parallel computation**: Self-attention processes all tokens simultaneously, unlike RNNs. This enables efficient training on modern hardware.
2. **Long-range dependencies**: Self-attention connects any two positions with a constant path length ($O(1)$ operations), compared to $O(n)$ for RNNs.
3. **Multi-head representations**: Different heads can learn different types of relationships (local syntax, long-range semantics, etc.).
4. **Deep stacking**: Residual connections and layer norm enable very deep models.

## Complexity

| Component | Per-Layer Complexity | Sequential Ops | Max Path Length |
|---|---|---|---|
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d)$ | $O(1)$ | $O(\log_k(n))$ |

The quadratic $O(n^2)$ complexity of self-attention with respect to sequence length is its main drawback, motivating efficient variants (Linformer, Longformer, Reformer).

## Training Details

The Transformer is trained with the **Adam optimizer** ($\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$) and a custom learning rate schedule that first warms up linearly for 4000 steps, then decays proportionally to the inverse square root of the step number:

$$\text{lr}(n) = d_{\text{model}}^{-0.5} \cdot \min(n^{-0.5}, n \cdot 4000^{-1.5})$$

This schedule is critical — the warmup phase prevents early training instability (since the model's parameters are random, large updates from attention gradients would destabilize training), while the decay phase allows fine-grained convergence.

**Label smoothing** of $\epsilon_{ls} = 0.1$ is applied to the target distribution, replacing hard one-hot targets with a mixture of the target token ($1 - \epsilon_{ls}$) and a uniform distribution over the vocabulary ($\epsilon_{ls} / V$). This prevents the model from becoming over-confident and improves both BLEU score and model calibration.

**Dropout** is applied to the output of each sub-layer before the residual connection is added, with a dropout rate of 0.1. This regularizes the large model and prevents co-adaptation between attention heads.

## Inference: Autoregressive Generation

At inference time, the encoder processes the input sequence once, and the decoder generates tokens one at a time using a left-to-right beam search (typically beam size = 4). The decoder's self-attention at step $t$ only attends to its own previously generated tokens (positions $0$ through $t-1$), using a look-ahead mask. Unlike training, no teacher forcing is available — the decoder must condition on its own predictions, making beam search essential for high-quality outputs.

## Why the Transformer Works
