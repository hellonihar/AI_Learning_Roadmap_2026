# 05 — Weight Initialization

Proper weight initialization is crucial for training deep networks. Poor initialization can cause **vanishing** or **exploding** gradients, making learning impossible.

---

## Why Initialization Matters

Consider a forward pass through $L$ layers with linear activations (for analysis):

$$\mathbf{a}^{[L]} = \mathbf{W}^{[L]} \mathbf{W}^{[L-1]} \cdots \mathbf{W}^{[1]} \mathbf{x}$$

If weights are too large, activations explode; if too small, they vanish. The same applies to gradients during backpropagation. The goal of initialization is to **keep the variance of activations and gradients stable** across layers.

---

## 1. Zero Initialization

Setting all weights to zero.

$$W^{[l]}_{ij} = 0 \quad \forall \, i, j, l$$

**Problem:** Every neuron in a layer computes the same output, and all gradients are identical. The network never breaks symmetry and cannot learn different features.

**Verdict:** Never use.

---

## 2. Random Initialization (Small)

$$W^{[l]}_{ij} \sim \mathcal{U}(-\epsilon, \epsilon) \quad \text{or} \quad \mathcal{N}(0, \epsilon^2)$$

Typically $\epsilon = 0.01$.

| Pros | Cons |
|---|---|
| Simple, breaks symmetry | For deep networks, small values cause vanishing gradients |
| Works for shallow nets | Large values cause exploding gradients |

**Verdict:** Use only for very shallow networks.

---

## 3. Xavier / Glorot Initialization

Designed for activations like **tanh** or **sigmoid** that are symmetric around zero.

**Uniform version:**

$$W^{[l]}_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$

**Normal version:**

$$W^{[l]}_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$

where $n_{\text{in}}$ is the number of input units and $n_{\text{out}}$ the number of output units.

**Idea:** Keeps the variance of activations the same across layers. Derivation assumes the activation function is approximately linear near 0.

**Use when:** Tanh / sigmoid / softmax layers.

---

## 4. He Initialization (Kaiming)

Designed for **ReLU** and its variants, which are not symmetric and zero out half the activations.

$$W^{[l]}_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

**Why $2/n_{\text{in}}$?** Since ReLU zeros half the activations, the variance is halved. Doubling the variance of the weights compensates for this.

**Use when:** ReLU, Leaky ReLU, PReLU.

---

## Comparison Summary

| Initialization | Distribution | Variance | Best For |
|---|---|---|---|
| Zero | Constant 0 | 0 | ❌ Avoid |
| Random small | $\mathcal{N}(0, 0.0001)$ | 0.0001 | Shallow nets only |
| Xavier/Glorot | $\mathcal{N}(0, \frac{2}{n_{\text{in}}+n_{\text{out}}})$ | $\frac{2}{n_{\text{in}}+n_{\text{out}}}$ | Tanh, Sigmoid, Softmax |
| He/Kaiming | $\mathcal{N}(0, \frac{2}{n_{\text{in}}})$ | $\frac{2}{n_{\text{in}}}$ | ReLU, Leaky ReLU |

---

## Practical Guidelines

- **Default choice:** He initialization with ReLU activations.
- **For tanh networks:** Use Xavier initialization.
- **Bias initialization:** Always initialize biases to 0 — they break symmetry independently via weight gradients.
- **Very deep networks ($>50$ layers):** Consider residual connections (ResNet) or layer-wise adaptive initialization.

---

**Next:** [06 - Vanishing & Exploding Gradients.md](./06%20-%20Vanishing%20%26%20Exploding%20Gradients.md)
