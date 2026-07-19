# Exercises: Seq2Seq Models & Attention

## Exercise 1: Manual Bahdanau Score Computation

Given:
- Decoder hidden state $s = [0.5, -0.2]$
- Encoder states $h_1 = [0.1, 0.3]$, $h_2 = [-0.4, 0.6]$, $h_3 = [0.7, -0.1]$
- $W_a = \begin{bmatrix} 0.5 & 0.3 \\ -0.2 & 0.8 \end{bmatrix}$, $U_a = \begin{bmatrix} 0.6 & -0.1 \\ 0.4 & 0.2 \end{bmatrix}$, $v_a = [0.3, 0.7]$

Compute the alignment scores $e_{t,i}$ for $i = 1, 2, 3$ and the attention weights $\alpha_{t,i}$.

## Exercise 2: Scaled Dot-Product Attention

Let $Q = \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}$, $K = \begin{bmatrix}0 & 1 \\ 1 & 0 \\ 1 & 1\end{bmatrix}$, $V = \begin{bmatrix}1 & 2 \\ 2 & 1 \\ 3 & 0\end{bmatrix}$, $d_k = 2$.

Compute:
1. The attention scores $S = QK^\top / \sqrt{d_k}$
2. The attention weights after softmax
3. The final attention output

## Exercise 3: Multi-Head Attention Dimensions

The Transformer base model has $d_{\text{model}} = 512$, $h = 8$ heads.

1. What are $d_k$ and $d_v$?
2. If the input sequence length is $n = 20$, what is the shape of $Q$, $K$, $V$ for a single head, and what is the shape of the attention scores?
3. After concatenating 8 heads, what is the output dimension? Why is this specific value chosen?

## Exercise 4: Prove the Derivative of Self-Attention

Let $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$.

Denote $S = QK^\top / \sqrt{d_k}$ and $P = \text{softmax}(S)$ (applied row-wise), so the output is $O = PV$.

Derive $\frac{\partial L}{\partial S}$ where $L$ is the scalar loss. Show that:

$$\frac{\partial L}{\partial S} = P \odot \left( \frac{\partial L}{\partial O} V^\top - \text{diag}(P^\top \frac{\partial L}{\partial O} V^\top) \right)$$

Hint: The softmax gradient for a single row $p$ with logits $s$ is $\frac{\partial p_j}{\partial s_i} = p_i(\delta_{ij} - p_j)$.

## Exercise 5: Compare Additive vs. Multiplicative Attention

Given the same hidden dimension $d_h$, compare the computational complexity (number of multiplications) of computing a single alignment score for:

1. Bahdanau (additive) attention
2. Luong (dot-product) attention
3. Luong (general) attention

Assume all weight matrices are $d_h \times d_h$ and $v_a$ is $d_h$.

## Exercise 6: Positional Encoding Frequencies

For $d_{\text{model}} = 4$, compute the positional encodings $PE_{(pos, 2i)}$ and $PE_{(pos, 2i+1)}$ for $pos = 0, 1, 2, 3$.

1. What is the wavelength for each dimension $i$?
2. For two positions $pos = 5$ and $pos = 10$, compute the encoding difference. Is it approximately linear?
3. Why does the Transformer use both sine and cosine functions instead of just one?

## Exercise 7: Masking Effects

Given a sequence of length 4 with padding token at position 3 (0-indexed):

1. Construct the combined padding + look-ahead mask for the decoder at training time.
2. Show what happens to the attention scores when a masked position's score is set to $-\infty$ before softmax.
3. For a sequence `[A, B, <PAD>, <PAD>]` with positional encodings, explain why the padding mask is necessary even though positional encodings already distinguish the positions.

## Exercise 8: BERT vs. GPT Design Decisions

1. BERT uses bidirectional context; GPT uses unidirectional. Why can't BERT be used for text generation?
2. GPT-3 can perform tasks with few-shot prompting, while BERT requires fine-tuning. What property of the autoregressive LM objective enables this?
3. Design a hybrid: How would you modify BERT to generate text? What attention mask would you use?

---

# Answers

## Answer 1

**Step 1: Compute $W_a s$**
$$W_a s = \begin{bmatrix} 0.5 & 0.3 \\ -0.2 & 0.8 \end{bmatrix} \begin{bmatrix} 0.5 \\ -0.2 \end{bmatrix} = \begin{bmatrix} 0.25 - 0.06 \\ -0.10 - 0.16 \end{bmatrix} = [0.19, -0.26]$$

**Step 2: Compute $U_a h_i$**

For $h_1 = [0.1, 0.3]$:
$$U_a h_1 = \begin{bmatrix} 0.6 & -0.1 \\ 0.4 & 0.2 \end{bmatrix} \begin{bmatrix} 0.1 \\ 0.3 \end{bmatrix} = \begin{bmatrix} 0.06 - 0.03 \\ 0.04 + 0.06 \end{bmatrix} = [0.03, 0.10]$$

For $h_2 = [-0.4, 0.6]$:
$$U_a h_2 = \begin{bmatrix} 0.6 & -0.1 \\ 0.4 & 0.2 \end{bmatrix} \begin{bmatrix} -0.4 \\ 0.6 \end{bmatrix} = \begin{bmatrix} -0.24 - 0.06 \\ -0.16 + 0.12 \end{bmatrix} = [-0.30, -0.04]$$

For $h_3 = [0.7, -0.1]$:
$$U_a h_3 = \begin{bmatrix} 0.6 & -0.1 \\ 0.4 & 0.2 \end{bmatrix} \begin{bmatrix} 0.7 \\ -0.1 \end{bmatrix} = \begin{bmatrix} 0.42 + 0.01 \\ 0.28 - 0.02 \end{bmatrix} = [0.43, 0.26]$$

**Step 3: Compute $\tanh(W_a s + U_a h_i)$**

$h_1: \tanh([0.22, -0.16]) = [0.2165, -0.1587]$
$h_2: \tanh([-0.11, -0.30]) = [-0.1096, -0.2913]$
$h_3: \tanh([0.62, 0.00]) = [0.5514, 0.0000]$

**Step 4: Compute $v_a^\top \tanh(\dots)$**

$e_1 = 0.3 \cdot 0.2165 + 0.7 \cdot (-0.1587) = 0.0650 - 0.1111 = -0.0461$
$e_2 = 0.3 \cdot (-0.1096) + 0.7 \cdot (-0.2913) = -0.0329 - 0.2039 = -0.2368$
$e_3 = 0.3 \cdot 0.5514 + 0.7 \cdot 0.0000 = 0.1654 + 0.0000 = 0.1654$

**Step 5: Softmax**

$\alpha_1 = e^{-0.0461} / (e^{-0.0461} + e^{-0.2368} + e^{0.1654}) = 0.955 / (0.955 + 0.789 + 1.180) = 0.327$
$\alpha_2 = 0.789 / 2.924 = 0.270$
$\alpha_3 = 1.180 / 2.924 = 0.403$

## Answer 2

**Scores**:
$$S = \frac{1}{\sqrt{2}} \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix} \begin{bmatrix}0 & 1 & 1 \\ 1 & 0 & 1\end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix}0 & 1 & 1 \\ 1 & 0 & 1\end{bmatrix} = \begin{bmatrix}0 & 0.707 & 0.707 \\ 0.707 & 0 & 0.707\end{bmatrix}$$

**Softmax weights** (row-wise):
Row 0: $[e^0, e^{0.707}, e^{0.707}] = [1, 2.028, 2.028] \to [1/5.056, 2.028/5.056, 2.028/5.056] = [0.198, 0.401, 0.401]$
Row 1: symmetric = $[0.401, 0.198, 0.401]$

**Output**:
Row 0: $0.198[1,2] + 0.401[2,1] + 0.401[3,0] = [0.198+0.802+1.203, 0.396+0.401+0] = [2.203, 0.797]$
Row 1: $0.401[1,2] + 0.198[2,1] + 0.401[3,0] = [0.401+0.396+1.203, 0.802+0.198+0] = [2.000, 1.000]$

## Answer 3

1. $d_k = d_v = 512/8 = 64$.
2. For a single head: $Q \in \mathbb{R}^{20 \times 64}$, $K \in \mathbb{R}^{20 \times 64}$, $V \in \mathbb{R}^{20 \times 64}$. Scores: $20 \times 20$.
3. Concatenated output: $20 \times (8 \cdot 64) = 20 \times 512$. This matches $d_{\text{model}}$ so the residual connection can add the original input directly.

## Answer 4

Let $O = PV$ where $P = \text{softmax}(S)$ row-wise.

For a single row $p$ and corresponding $o = pV$:
$$\frac{\partial L}{\partial s_i} = \sum_j \frac{\partial L}{\partial o_j} \frac{\partial o_j}{\partial s_i}$$

Since $\frac{\partial o_j}{\partial s_i} = \sum_k \frac{\partial p_k}{\partial s_i} V_{kj}$ and $\frac{\partial p_k}{\partial s_i} = p_k(\delta_{ik} - p_i)$:

$$\frac{\partial L}{\partial s_i} = \sum_j \frac{\partial L}{\partial o_j} \sum_k p_k(\delta_{ik} - p_i) V_{kj} = p_i \sum_j \frac{\partial L}{\partial o_j} V_{ij} - p_i \sum_k p_k \sum_j \frac{\partial L}{\partial o_j} V_{kj}$$

The first term is $p_i \cdot ( \frac{\partial L}{\partial O} V^\top)_{ii}$ and the second is $p_i \cdot (P^\top \frac{\partial L}{\partial O} V^\top)_{ii}$. In matrix form:

$$\frac{\partial L}{\partial S} = P \odot \left( \frac{\partial L}{\partial O} V^\top - \mathbf{1} \cdot \text{diag}(P^\top \frac{\partial L}{\partial O} V^\top)^\top \right)$$

where $\odot$ is element-wise multiplication.

## Answer 5

| Variant | Ops per score |
|---|---|
| Bahdanau | $W_a s$: $d_h^2$ + $U_a h_i$: $d_h^2$ + add: $d_h$ + $\tanh$: $d_h$ + $v_a^\top(\cdot)$: $d_h$ = $2d_h^2 + 3d_h$ |
| Luong dot | $s^\top h_i$: $d_h$ multiplications |
| Luong general | $W_a s$: $d_h^2$ + $(W_a s)^\top h_i$: $d_h$ = $d_h^2 + d_h$ |

For $d_h = 512$: Bahdanau ~524k ops, general ~262k ops, dot ~512 ops.

## Answer 6

For $d_{\text{model}} = 4$: $i = 0, 1$.

$i=0$: $\omega_0 = 1/10000^{0/4} = 1$. $PE_{(pos, 0)} = \sin(pos)$, $PE_{(pos, 1)} = \cos(pos)$.
$i=1$: $\omega_1 = 1/10000^{2/4} = 1/100$. $PE_{(pos, 2)} = \sin(pos/100)$, $PE_{(pos, 3)} = \cos(pos/100)$.

| pos | dim 0 | dim 1 | dim 2 | dim 3 |
|---|---|---|---|---|
| 0 | 0.00 | 1.00 | 0.00 | 1.00 |
| 1 | 0.84 | 0.54 | 0.01 | 1.00 |
| 2 | 0.91 | -0.42 | 0.02 | 1.00 |
| 3 | 0.14 | -0.99 | 0.03 | 1.00 |

1. Wavelength for $i=0$: $2\pi$ (~6.28). For $i=1$: $2\pi \cdot 100 \approx 628$.
2. The difference for dim 0 between pos 5 and 10 is $\sin(10) - \sin(5) \approx -0.544 - (-0.959) = 0.415$. This is not precisely linear but linear approximations hold locally due to the $\sin(pos+k)$ formula.
3. Both sine and cosine enable the model to represent any offset $k$ as a linear transformation: $\sin(pos+k) = \sin(pos)\cos(k) + \cos(pos)\sin(k)$. This linearity helps the model learn relative position patterns.

## Answer 7

1. Sequence `[w0, w1, w2, <PAD>]` (length 4, pad at index 3).
   Padding mask: `[[0,0,0,1]]` (1 = masked).
   Look-ahead mask: `[[0,1,1,1],[0,0,1,1],[0,0,0,1],[0,0,0,0]]` (1 = masked, upper triangle).
   Combined: OR the two masks. For decoder step $t$ and position $j$, masked if $j > t$ OR $j = 3$.

2. Example for decoder step 1 (attending to `[w0, w1, w2, PAD]`): scores `[2.1, 1.5, 0.8, 0.3]`. After masking: `[2.1, 1.5, 0.8, -inf]`. Softmax: $[e^{2.1}/(e^{2.1}+e^{1.5}+e^{0.8}), e^{1.5}/(...), e^{0.8}/(...), 0]$. The masked position gets exactly zero attention.

3. Even with distinct positional encodings, the attention mechanism would still compute non-zero similarity between the query and the `<PAD>` token's position. The model would waste capacity attending to padding and might learn spurious alignments. The mask enforces zero attention, making the model ignore padding entirely.

## Answer 8

1. BERT uses bidirectional attention — each token attends to all others including future tokens. For generation (predicting the next token), this would leak information about what comes next, making the prediction trivial. Text generation requires causal (left-to-right) masking.

2. The autoregressive LM objective naturally aligns with the format of many tasks: given context (prompt), generate continuation. GPT learns to condition on arbitrary preceding text. In-context learning exploits this by treating task examples as prefixes the model must continue. BERT's MLM objective predicts masked tokens within a fixed context, which does not naturally generalize to generation tasks.

3. To make BERT generate, modify the attention mask to be causal (lower-triangular), converting the encoder into a decoder. This is essentially GPT's approach. Alternatively, add a decoder on top of BERT's encoder and use cross-attention (like T5).
