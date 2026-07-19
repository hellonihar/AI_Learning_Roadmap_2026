# 06 — Vanishing & Exploding Gradients

As networks grow deeper, gradients can become exponentially small (vanish) or exponentially large (explode) as they are backpropagated through many layers. This makes training deep networks extremely difficult.

---

## 1. The Vanishing Gradient Problem

### 1.1 Cause

During backpropagation, the gradient at layer $l$ contains a product of Jacobian matrices:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \cdot \left( \prod_{k=l+1}^{L} \mathbf{J}_k \right) \cdot \frac{\partial \mathbf{a}^{[l]}}{\partial \mathbf{W}^{[l]}}$$

where $\mathbf{J}_k = \frac{\partial \mathbf{a}^{[k]}}{\partial \mathbf{a}^{[k-1]}} = \text{diag}\left( g'(\mathbf{z}^{[k]}) \right) \cdot \mathbf{W}^{[k]^\top}$.

If the eigenvalues of most $\mathbf{J}_k$ are less than $1$, the product **shrinks exponentially** with depth.

### 1.2 Role of Sigmoid/Tanh

Sigmoid and tanh saturate for large $|z|$:

$$\sigma'(z) = \sigma(z)(1 - \sigma(z)) \leq 0.25$$

$$\tanh'(z) = 1 - \tanh^2(z) \leq 1.0$$

When multiplied across layers, these small derivatives cause gradients to **approach zero** in early layers. The network's early layers learn extremely slowly or not at all.

### 1.3 Symptoms

- Early layer weights barely change during training.
- Training loss decreases very slowly or stalls.
- Deeper layers learn faster than earlier ones.

---

## 2. The Exploding Gradient Problem

### 2.1 Cause

If the eigenvalues of the Jacobian products exceed $1$, gradients **grow exponentially** with depth. This is common when:

- Weights are initialized too large.
- Using unbounded activations (ReLU) without proper initialization.
- Recurrent networks (vanilla RNNs) process long sequences.

### 2.2 Symptoms

- Loss becomes `NaN` or `Inf` during training.
- Weights grow very large after a few updates.
- Oscillating or diverging loss curve.

---

## 3. Solutions

### 3.1 Use Non-Saturating Activations

**ReLU** and its variants have a gradient of $1$ for $z > 0$, which does not shrink the gradient. This is the single most impactful fix for vanishing gradients.

### 3.2 Proper Weight Initialization

- **He initialization** for ReLU networks.
- **Xavier initialization** for tanh networks.
- Keeps the variance of activations and gradients stable across layers.

### 3.3 Batch Normalization

Introduced by Ioffe & Szegedy (2015). Normalizes each layer's pre-activations to have zero mean and unit variance:

$$\hat{z}^{(i)} = \frac{z^{(i)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$z_{\text{norm}}^{(i)} = \gamma \hat{z}^{(i)} + \beta$$

- Prevents activations from saturating (keeps them in the linear regime of the activation).
- Allows much higher learning rates.
- Adds slight regularization (noise from mini-batch statistics).

### 3.4 Residual Connections (ResNet)

Introduced by He et al. (2015). Adds a skip connection around each layer block:

$$\mathbf{a}^{[l+1]} = \mathcal{F}(\mathbf{a}^{[l]}) + \mathbf{a}^{[l]}$$

- Gradients can flow directly through the **identity path**, bypassing the layer.
- This allows training networks with **hundreds or thousands** of layers.
- The Jacobian of the identity is the identity matrix — no shrinkage.

### 3.5 Gradient Clipping

Used primarily in RNNs to prevent exploding gradients:

$$\mathbf{g} \leftarrow \begin{cases} \mathbf{g} \cdot \frac{\text{clip\_norm}}{\|\mathbf{g}\|} & \text{if } \|\mathbf{g}\| > \text{clip\_norm} \\ \mathbf{g} & \text{otherwise} \end{cases}$$

- Simple to implement ($\text{clip\_norm}$ is usually $1.0$ or $5.0$).
- Preserves gradient direction while preventing overflow.

### 3.6 Other Techniques

- **LSTM/GRU** — gating mechanisms solve vanishing gradients in RNNs.
- **Careful learning rate scheduling** — warmup + decay helps early stability.
- **Layer normalization** — alternative to batch norm for RNNs/transformers.

---

## Comparison of Solutions

| Technique | Vanishing | Exploding | Complexity |
|---|---|---|---|
| ReLU activation | ✅ Strong | ❌ May worsen | Very low |
| He/Xavier init | ✅ | ✅ | Very low |
| Batch norm | ✅ | ✅ | Medium |
| Residual connections | ✅ Strong | ✅ | Low |
| Gradient clipping | ❌ | ✅ Strong | Very low |

---

## Key Takeaway

Modern deep learning relies on a **combination** of techniques: ReLU + He initialization handles most vanishing gradient issues; batch normalization and residual connections enable networks deeper than $50$ layers; gradient clipping catches exploding gradients in RNNs.

---

**Next:** [code_walkthrough.md](./code_walkthrough.md)
