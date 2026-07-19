# Exercises: Sequence Models

## Exercise 1: Derive LSTM Gate Gradients

Derive the gradient of the loss $\mathcal{L}$ with respect to the forget gate bias $\mathbf{b}_f$ in an LSTM. Show the full chain rule expansion.

<details>
<summary>Answer</summary>

For the forget gate $\mathbf{f}_t = \sigma(\mathbf{W}_{hf} \mathbf{h}_{t-1} + \mathbf{W}_{xf} \mathbf{x}_t + \mathbf{b}_f)$, let $\mathbf{a}_f = \mathbf{W}_{hf} \mathbf{h}_{t-1} + \mathbf{W}_{xf} \mathbf{x}_t + \mathbf{b}_f$.

By the chain rule:
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_f} = \frac{\partial \mathcal{L}}{\partial \mathbf{f}_t} \cdot \frac{\partial \mathbf{f}_t}{\partial \mathbf{a}_f} \cdot \frac{\partial \mathbf{a}_f}{\partial \mathbf{b}_f}
$$

Where:
- $\frac{\partial \mathbf{f}_t}{\partial \mathbf{a}_f} = \text{diag}(\mathbf{f}_t \odot (1 - \mathbf{f}_t))$ (sigmoid derivative)
- $\frac{\partial \mathbf{a}_f}{\partial \mathbf{b}_f} = \mathbf{I}$ (identity)
- $\frac{\partial \mathcal{L}}{\partial \mathbf{f}_t} = \frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} \odot \mathbf{c}_{t-1}$ (since $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \cdots$)
- $\frac{\partial \mathcal{L}}{\partial \mathbf{c}_t}$ comes from backpropagation through future time steps

Thus:
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_f} = \left( \frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} \odot \mathbf{c}_{t-1} \odot \mathbf{f}_t \odot (1 - \mathbf{f}_t) \right)
$$
</details>

---

## Exercise 2: Implement GRU Forward Pass

Write the forward pass equations for a GRU. Given $\mathbf{x}_t = [0.5, -0.2]^T$ and $\mathbf{h}_{t-1} = [0.1, 0.3]^T$ with the following weights:

- $\mathbf{W}_{xr} = [[0.1, 0.2], [0.3, 0.4]]$, $\mathbf{W}_{hr} = [[0.5, 0.6], [0.7, 0.8]]$, $\mathbf{b}_r = [0.01, 0.02]$
- $\mathbf{W}_{xz} = [[0.2, 0.3], [0.4, 0.5]]$, $\mathbf{W}_{hz} = [[0.6, 0.7], [0.8, 0.9]]$, $\mathbf{b}_z = [0.03, 0.04]$
- $\mathbf{W}_{xh} = [[0.3, 0.4], [0.5, 0.6]]$, $\mathbf{W}_{hh} = [[0.7, 0.8], [0.9, 1.0]]$, $\mathbf{b}_h = [0.05, 0.06]$

Compute $\mathbf{r}_t$, $\mathbf{z}_t$, $\tilde{\mathbf{h}}_t$, and $\mathbf{h}_t$.

<details>
<summary>Answer</summary>

Step 1: Compute the reset gate:
$$
\mathbf{a}_r = \mathbf{W}_{xr} \mathbf{x}_t + \mathbf{W}_{hr} \mathbf{h}_{t-1} + \mathbf{b}_r
$$
$\mathbf{W}_{xr} \mathbf{x}_t = [0.1 \cdot 0.5 + 0.2 \cdot (-0.2), 0.3 \cdot 0.5 + 0.4 \cdot (-0.2)] = [0.01, 0.07]$
$\mathbf{W}_{hr} \mathbf{h}_{t-1} = [0.5 \cdot 0.1 + 0.6 \cdot 0.3, 0.7 \cdot 0.1 + 0.8 \cdot 0.3] = [0.23, 0.31]$
$\mathbf{a}_r = [0.01 + 0.23 + 0.01, 0.07 + 0.31 + 0.02] = [0.25, 0.40]$
$\mathbf{r}_t = \sigma([0.25, 0.40]) = [0.562, 0.599]$

Step 2: Compute the update gate:
$\mathbf{W}_{xz} \mathbf{x}_t = [0.2 \cdot 0.5 + 0.3 \cdot (-0.2), 0.4 \cdot 0.5 + 0.5 \cdot (-0.2)] = [0.04, 0.10]$
$\mathbf{W}_{hz} \mathbf{h}_{t-1} = [0.6 \cdot 0.1 + 0.7 \cdot 0.3, 0.8 \cdot 0.1 + 0.9 \cdot 0.3] = [0.27, 0.35]$
$\mathbf{a}_z = [0.04 + 0.27 + 0.03, 0.10 + 0.35 + 0.04] = [0.34, 0.49]$
$\mathbf{z}_t = \sigma([0.34, 0.49]) = [0.584, 0.620]$

Step 3: Compute candidate hidden state:
$\mathbf{r}_t \odot \mathbf{h}_{t-1} = [0.562 \cdot 0.1, 0.599 \cdot 0.3] = [0.056, 0.180]$
$\mathbf{W}_{xh} \mathbf{x}_t = [0.3 \cdot 0.5 + 0.4 \cdot (-0.2), 0.5 \cdot 0.5 + 0.6 \cdot (-0.2)] = [0.07, 0.13]$
$\mathbf{W}_{hh}(\mathbf{r}_t \odot \mathbf{h}_{t-1}) = [0.7 \cdot 0.056 + 0.8 \cdot 0.180, 0.9 \cdot 0.056 + 1.0 \cdot 0.180] = [0.183, 0.230]$
$\mathbf{a}_h = [0.07 + 0.183 + 0.05, 0.13 + 0.230 + 0.06] = [0.303, 0.420]$
$\tilde{\mathbf{h}}_t = \tanh([0.303, 0.420]) = [0.294, 0.397]$

Step 4: Final hidden state:
$\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t$
$\mathbf{h}_t = [0.584 \cdot 0.1 + 0.416 \cdot 0.294, 0.620 \cdot 0.3 + 0.380 \cdot 0.397]$
$\mathbf{h}_t = [0.058 + 0.122, 0.186 + 0.151] = [0.180, 0.337]$
</details>

---

## Exercise 3: Compare RNN, LSTM, GRU on a Toy Task

Describe a controlled experiment comparing vanilla RNN, LSTM, and GRU on the **adding problem**: given a sequence of random numbers with two marked positions, the model must output the sum of the two marked numbers. Sequence lengths: 10, 50, 100, 200. Which model succeeds at each length and why?

<details>
<summary>Answer</summary>

- **Vanilla RNN**: Fails at length 50+ due to vanishing gradients. The error signal cannot propagate through 50+ multiplications of $\tanh' \cdot \mathbf{W}_{hh}$.
- **LSTM**: Succeeds at all lengths. The constant error carousel allows gradients to flow unchanged through the cell state when the forget gate is open.
- **GRU**: Succeeds at lengths 10–100, may degrade slightly at 200. The update gate provides a linear shortcut for gradients, but the unified hidden state is more susceptible to interference than the LSTM's separate cell state.

The adding problem is the canonical test for long-range dependency learning. LSTMs were specifically designed to solve it.
</details>

---

## Exercise 4: Why is BPTT Expensive?

Explain the computational and memory complexity of BPTT for a sequence of length $T$ with hidden size $d_h$. Compare to standard backpropagation in a feedforward network of $T$ layers.

<details>
<summary>Answer</summary>

**Time complexity**: $O(T \cdot d_h^2)$ for both forward and backward passes. The quadratic term comes from the matrix-vector multiply $\mathbf{W}_{hh} \mathbf{h}_{t-1}$.

**Memory complexity**: $O(T \cdot d_h)$. All hidden states $\mathbf{h}_1, \dots, \mathbf{h}_T$ must be stored during the forward pass because they are needed for gradient computation in the backward pass.

Compared to a feedforward network of $T$ layers:
- Similar time complexity per layer, but RNN layers share parameters (same $\mathbf{W}_{hh}$ at each step).
- BPTT is more memory-intensive because the full unrolled graph must be materialized.

**Why BPTT is more expensive than standard backprop**: The unrolled graph has $T$ sequential dependencies. In a feedforward network, all layers can be computed in one pass. In an RNN, the forward pass is inherently sequential, and the backward pass requires traversing all $T$ steps in reverse, making it impossible to parallelize across time steps.
</details>

---

## Exercise 5: Gradient Flow Visualization

For a vanilla RNN, derive the expression for $\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1}$ in terms of products of Jacobians. Explain how this product leads to vanishing gradients when $T$ is large.

<details>
<summary>Answer</summary>

By repeated application of the chain rule through time:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \prod_{t=2}^{T} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}
$$

Where each Jacobian is:
$$
\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \text{diag}\left(1 - \tanh^2(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)\right) \mathbf{W}_{hh}
$$

Let $\mathbf{D}_t$ be the diagonal matrix of $\tanh'$ values. Then:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \mathbf{D}_T \mathbf{W}_{hh} \mathbf{D}_{T-1} \mathbf{W}_{hh} \cdots \mathbf{D}_2 \mathbf{W}_{hh}
$$

With $T-1$ multiplications by $\mathbf{W}_{hh}$. If the spectral radius $\rho(\mathbf{W}_{hh}) < 1$, the product vanishes exponentially as $T$ grows. Since each $\mathbf{D}_t$ has entries in $[0, 1]$, the shrinkage is compounded.
</details>

---

## Exercise 6: Hyperparameter Analysis

Compare the number of parameters in a single-layer LSTM vs. a single-layer GRU for input size $d_x = 100$ and hidden size $d_h = 256$. Which has more and by how much?

<details>
<summary>Answer</summary>

**LSTM parameters** (4 sets of gates):
$$
4 \times (d_h d_x + d_h^2 + d_h) = 4 \times (256 \cdot 100 + 256^2 + 256)
$$
$$
= 4 \times (25,600 + 65,536 + 256) = 4 \times 91,392 = 365,568
$$

**GRU parameters** (3 sets of gates):
$$
3 \times (d_h d_x + d_h^2 + d_h) = 3 \times (25,600 + 65,536 + 256) = 3 \times 91,392 = 274,176
$$

LSTM has **91,392 more parameters** (33% more) for the same hidden size.
</details>

---

## Exercise 7: Design Choice — Separate Cell State

Explain why the LSTM's separate cell state $\mathbf{c}_t$ is more robust for very long sequences than the GRU's single hidden state $\mathbf{h}_t$.

<details>
<summary>Answer</summary>

In the LSTM, the cell state $\mathbf{c}_t$ interacts only through linear operations (addition and element-wise multiplication):
$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$

This means the gradient of $\mathbf{c}_t$ with respect to $\mathbf{c}_{t-1}$ is simply $\mathbf{f}_t$. If the forget gate is open ($\mathbf{f}_t \approx 1$), the gradient is approximately the identity, allowing error signals to propagate thousands of steps.

In the GRU, the hidden state update is:
$$
\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t
$$

While this also has a linear shortcut via $\mathbf{z}_t$, the hidden state participates in the candidate computation $\tilde{\mathbf{h}}_t$ (through the reset gate), creating additional nonlinear pathways that can interfere with gradient flow. The LSTM's explicit separation of memory ($\mathbf{c}_t$) from output ($\mathbf{h}_t$) provides cleaner gradient highways.

Empirically, both handle sequences up to ~200 steps well, but LSTM tends to be more reliable for 500+ step dependencies.
</details>

---

## Exercise 8: Gradient Clipping

Explain why gradient clipping is necessary for RNNs but less critical for deep feedforward networks. What happens if you clip too aggressively?

<details>
<summary>Answer</summary>

**Why necessary for RNNs**: The BPTT gradient involves repeated multiplication by $\mathbf{W}_{hh}$. If the spectral radius $\rho(\mathbf{W}_{hh}) > 1$, gradients grow exponentially with sequence length $T$. This is unique to RNNs because the same weight matrix is applied at each time step, creating a dynamical system that can be unstable.

In feedforward networks, each layer has different weight matrices, and careful initialization (e.g., He or Xavier) keeps gradients in a reasonable range. The exploding gradient problem is far more severe in RNNs.

**Overly aggressive clipping**: If the clip threshold is too small, the gradient no longer points in the true direction of steepest descent, slowing or preventing convergence. The model may fail to learn long-range dependencies because large (but meaningful) gradients are truncated.
</details>
