# Attention Mechanism

## The Information Bottleneck Problem

The classic encoder-decoder compresses the entire input sequence into a single fixed-length context vector $c$. This creates an **information bottleneck**: long sequences lose detail, and the decoder must extract all relevant information from one vector. Bahdanau et al. (2015) proposed **attention** as a solution — instead of a single context vector, the decoder computes a new context vector at each time step by focusing on the most relevant parts of the input.

## Bahdanau (Additive) Attention

Bahdanau attention, also called **additive attention** or **concatenative attention**, computes a context vector $c_t$ for each decoder time step $t$ as a weighted sum of the encoder's hidden states $h_1, h_2, \dots, h_T$.

### Alignment Scores

For each encoder hidden state $h_i$, we compute an **alignment score** $e_{t,i}$ that measures how well the decoder's current hidden state $s_{t-1}$ (or $s_t$ depending on formulation) matches $h_i$:

$$e_{t,i} = v_a^\top \tanh(W_a s_{t-1} + U_a h_i)$$

where:
- $v_a$, $W_a$, $U_a$ are learned weight matrices and vectors
- $s_{t-1}$ is the previous decoder hidden state
- $h_i$ is the $i$-th encoder hidden state

The additive combination of $s_{t-1}$ and $h_i$ through a $\tanh$ non-linearity allows the model to learn complex alignments. The weight matrices $W_a$ and $U_a$ project both states into a common attention space of dimension $d_{\text{attn}}$, and $v_a$ then scores each projected combination. This is equivalent to a single-layer feed-forward network with a $\tanh$ activation, giving Bahdanau attention the capacity to learn non-linear alignments that multiplicative variants cannot express.

### Decoder State Initialization with Attention

In Bahdanau's formulation, the encoder is bidirectional, and the decoder initial hidden state $s_0$ is computed from a separate transformation of the backward encoder's final state, rather than being set to the context vector directly. This decouples the initial decoder state from the attention mechanism, allowing the model to learn when to rely on attention versus the recurrent state. The decoder's weight matrices are also initialized to be larger than typical RNN weights, which empirically improves gradient flow in the early stages of training.

### Attention Weights

The alignment scores are normalized across all encoder positions using softmax, producing the **attention weight** (also called alignment probability) $\alpha_{t,i}$:

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{k=1}^{T} \exp(e_{t,k})}$$

These weights form an $T' \times T$ matrix (decoder steps $\times$ encoder steps) called the **attention matrix**, where each row sums to 1.

### Context Vector

The context vector $c_t$ for decoder step $t$ is the weighted sum of all encoder hidden states:

$$c_t = \sum_{i=1}^{T} \alpha_{t,i} h_i$$

This allows the decoder to "attend" to different parts of the input at each generation step. For example, in translation, when generating the word "student", the model might assign high attention weight to the encoder hidden state corresponding to "étudiant".

### Decoder with Attention

The context vector $c_t$ is combined with the decoder state to produce the output:

$$s_t = g(y_{t-1}, s_{t-1}, c_t)$$
$$P(y_t \mid y_{<t}, \mathbf{x}) = \text{softmax}(W_o [s_t; c_t] + b_o)$$

## Luong (Multiplicative) Attention

Luong et al. (2015) proposed a family of simpler, often more efficient attention mechanisms. The key difference is when attention is computed relative to the decoder step.

### Score Functions

Luong defines three scoring variants for computing $e_{t,i}$:

**Dot product** (simplest):
$$e_{t,i} = s_t^\top h_i$$

**General (multiplicative)**:
$$e_{t,i} = s_t^\top W_a h_i$$

**Concat (additive, similar to Bahdanau)**:
$$e_{t,i} = v_a^\top \tanh(W_a [s_t; h_i])$$

The **dot** and **general** variants are called **multiplicative attention** because the score is computed via matrix multiplication rather than a neural network with a hidden layer.

### Global vs. Local Attention

Luong also distinguishes:

- **Global attention**: attends over all $T$ encoder hidden states (same as Bahdanau).
- **Local attention**: first predicts a single aligned position $p_t$ for the current decoder step, then computes attention over a window $[p_t - D, p_t + D]$ of encoder states. This reduces computational cost from $O(T)$ to $O(2D)$ where $D$ is a fixed window size.

## Comparison: Bahdanau vs. Luong

| Aspect | Bahdanau (Additive) | Luong (Multiplicative) |
|---|---|---|
| Score computation | $v_a^\top \tanh(W_a s_{t-1} + U_a h_i)$ | $s_t^\top h_i$ or $s_t^\top W_a h_i$ |
| Decoder state used | $s_{t-1}$ (previous) | $s_t$ (current) |
| Computational cost | Higher (requires small NN) | Lower (simple matrix ops) |
| Alignment expressiveness | More flexible (non-linear) | Less flexible (linear interaction) |
| Typical use | RNN-based Seq2Seq | RNN-based Seq2Seq |

In practice, both perform similarly on many tasks, but Luong's dot-product variant is faster and simpler.

## Attention Weight Matrix Visualization

A key diagnostic tool is visualizing the attention weight matrix as a heatmap, with encoder positions on the x-axis and decoder positions on the y-axis. A well-trained translation model should show approximately diagonal alignment for related languages (e.g., English-French), demonstrating that the model has learned to align source and target words.

## Input Feeding Approach

Luong proposed an **input feeding** strategy where the attention context vector $c_t$ from the previous decoder step is fed as additional input to the current step. This gives the decoder a "memory" of which input positions it has already attended to, promoting coverage and reducing the chance of repeatedly attending to the same position or skipping important parts of the input. The input feeding approach is especially beneficial for long sequences where coverage tracking becomes critical.

## Attention in Training

With attention, the decoder computes a unique context vector $c_t$ at every time step. During backpropagation, gradients flow from the output at step $t$ through $c_t$ to every encoder hidden state $h_i$, weighted by $\alpha_{t,i}$. This means the gradient signal reaches all encoder positions directly, even for very long sequences — effectively eliminating the vanishing gradient problem for the encoder-decoder connection. The attention mechanism also introduces additional parameters ($W_a, U_a, v_a$) that must be trained jointly with the encoder and decoder.

## Why Attention Works

Attention solves three problems simultaneously:

1. **Bypasses the bottleneck**: The decoder can access all encoder hidden states directly, not just the final one.
2. **Provides interpretability**: Attention weights show which parts of the input influence each output token.
3. **Improves gradient flow**: The weighted sum creates a direct connection between decoder output and each encoder hidden state, mitigating vanishing gradients.

## Mathematical Summary

Bahdanau attention at decoder step $t$:

$$
\begin{aligned}
e_{t,i} &= v_a^\top \tanh(W_a s_{t-1} + U_a h_i) \\
\alpha_{t,i} &= \frac{\exp(e_{t,i})}{\sum_{k=1}^{T} \exp(e_{t,k})} \\
c_t &= \sum_{i=1}^{T} \alpha_{t,i} h_i
\end{aligned}
$$

Luong (dot-product) attention at decoder step $t$:

$$
\begin{aligned}
e_{t,i} &= s_t^\top h_i \\
\alpha_{t,i} &= \frac{\exp(e_{t,i})}{\sum_{k=1}^{T} \exp(e_{t,k})} \\
c_t &= \sum_{i=1}^{T} \alpha_{t,i} h_i
\end{aligned}
$$

The attention mechanism was the key insight that led directly to the Transformer architecture, which replaces RNNs entirely with attention operations.
