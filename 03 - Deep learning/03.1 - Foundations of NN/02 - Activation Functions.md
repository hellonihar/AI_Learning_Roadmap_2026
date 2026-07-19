# 02 — Activation Functions

Activation functions introduce **non-linearity** into neural networks. Without them, stacking linear layers would collapse into a single linear transformation, making deep networks useless.

---

## 1. Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

$$\sigma'(z) = \sigma(z) \, (1 - \sigma(z))$$

**Plot:** S-shaped curve mapping any real input to $(0, 1)$. Steepest near $z = 0$; flattens at extremes.

| Pros | Cons |
|---|---|
| Smooth, differentiable everywhere | **Vanishing gradient**: gradients near 0 for extreme $z$ values |
| Outputs bounded in $(0,1)$ — useful for probabilities | Not zero-centered — causes zig-zagging in optimization |
| | Soft saturation kills gradient flow in deep networks |

**Use when:** Binary classification output layer (probability).

---

## 2. Tanh (Hyperbolic Tangent)

$$\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$$

$$\tanh'(z) = 1 - \tanh^2(z)$$

**Plot:** S-shaped, maps to $(-1, 1)$. Zero-centered, which is an advantage over sigmoid.

| Pros | Cons |
|---|---|
| Zero-centered — faster convergence than sigmoid | Still suffers from **vanishing gradient** at extremes |
| Stronger gradients for mid-range inputs | Saturation for large $|z|$ |

**Use when:** Hidden layers in shallow networks or RNNs (though ReLU is usually preferred today).

---

## 3. ReLU (Rectified Linear Unit)

$$\text{ReLU}(z) = \max(0, z)$$

$$\text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$$

**Plot:** Linear for $z > 0$, zero for $z \leq 0$.

| Pros | Cons |
|---|---|
| Computationally cheap — just $\max(0, z)$ | **Dying ReLU**: neurons can become permanently inactive if $z < 0$ always |
| Non-saturating in positive half — alleviates vanishing gradient | Not differentiable at $z = 0$ (subgradient $0$ is used) |
| Promotes sparse activation | Unbounded output can cause exploding activations |

**Use when:** Default for hidden layers in most feedforward and ConvNets.

---

## 4. Leaky ReLU

$$\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases}$$

where $\alpha \in (0, 1)$, typically $0.01$.

$$\text{LeakyReLU}'(z) = \begin{cases} 1 & z > 0 \\ \alpha & z \leq 0 \end{cases}$$

| Pros | Cons |
|---|---|
| Prevents dying ReLU — small gradient for $z < 0$ | Slightly more computation than ReLU |
| Retains benefits of ReLU for $z > 0$ | Performance gain over ReLU is often marginal |

**Use when:** Dead neurons are observed with ReLU.

---

## 5. ELU (Exponential Linear Unit)

$$\text{ELU}(z) = \begin{cases} z & z > 0 \\ \alpha (e^{z} - 1) & z \leq 0 \end{cases}$$

$$\text{ELU}'(z) = \begin{cases} 1 & z > 0 \\ \text{ELU}(z) + \alpha & z \leq 0 \end{cases}$$

| Pros | Cons |
|---|---|
| Smooth for all $z$ (differentiable everywhere) | Exponentiation is computationally expensive |
| Negative values push mean activation toward zero | Slower training than ReLU |
| Saturation in negative region provides noise robustness | |

**Use when:** Deeper networks where noise robustness matters.

---

## 6. Swish

$$\text{Swish}(z) = z \cdot \sigma(\beta z)$$

where $\sigma$ is the sigmoid function and $\beta$ is a learnable or fixed parameter (often $\beta = 1$).

$$\text{Swish}'(z) = \sigma(\beta z) + \beta z \cdot \sigma(\beta z) \cdot (1 - \sigma(\beta z))$$

| Pros | Cons |
|---|---|
| Smooth, non-monotonic — preserves negative information | More expensive to compute |
| Often outperforms ReLU on deeper networks (Google, 2017) | Less widely supported in inference engines |
| No dead neuron problem | |

**Use when:** Very deep networks where small accuracy gains justify extra compute.

---

## 7. GELU (Gaussian Error Linear Unit)

$$\text{GELU}(z) = z \cdot \Phi(z)$$

where $\Phi(z) = \frac{1}{2} \left[ 1 + \text{erf}\left( \frac{z}{\sqrt{2}} \right) \right]$ is the standard Gaussian CDF. A common approximation:

$$\text{GELU}(z) \approx 0.5z \left[ 1 + \tanh\left( \sqrt{\frac{2}{\pi}} (z + 0.044715 z^3) \right) \right]$$

| Pros | Cons |
|---|---|
| Smooth everywhere; stochastic regularization effect | Expensive to compute (approximation used in practice) |
| Used in **BERT, GPT, ViT** — performance proven | May overfit on very small datasets |
| Outperforms ReLU and Swish in many transformer models | |

**Use when:** Transformer architectures (NLP, vision transformers).

---

## Quick Reference

| Activation | Range | Output Zero-Centered? | Vanishing Gradient? | Compute Cost |
|---|---|---|---|---|
| Sigmoid | $(0, 1)$ | No | Yes | Medium |
| Tanh | $(-1, 1)$ | Yes | Yes | Medium |
| ReLU | $[0, \infty)$ | No (all $\geq 0$) | No | Very low |
| Leaky ReLU | $(-\infty, \infty)$ | Approx | No | Low |
| ELU | $(-\alpha, \infty)$ | Approx | No | High |
| Swish | $(-\infty, \infty)$ | No | No | High |
| GELU | $(-\infty, \infty)$ | No | No | High |

---

**Next:** [03 - Forward & Backward Propagation.md](./03%20-%20Forward%20%26%20Backward%20Propagation.md)
