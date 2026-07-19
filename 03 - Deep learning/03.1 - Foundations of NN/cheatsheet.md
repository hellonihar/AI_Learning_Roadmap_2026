# Cheatsheet — Foundations of Neural Networks

---

## Forward Pass (2-Layer Network)

| Step | Formula |
|---|---|
| Hidden pre-activation | $\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$ |
| Hidden activation | $\mathbf{a}_1 = g(\mathbf{z}_1)$ |
| Output pre-activation | $z_2 = \mathbf{W}_2 \mathbf{a}_1 + b_2$ |
| Output activation | $\hat{y} = a_2 = \sigma(z_2)$ |

---

## Backward Pass (2-Layer Network, BCE + Sigmoid Output)

| Step | Formula |
|---|---|
| Output error | $\delta_2 = \hat{y} - y$ |
| Output gradients | $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_2} = \delta_2 \mathbf{a}_1^\top$, $\frac{\partial \mathcal{L}}{\partial b_2} = \delta_2$ |
| Hidden error | $\delta_1 = (\mathbf{W}_2^\top \delta_2) \odot g'(\mathbf{z}_1)$ |
| Hidden gradients | $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \delta_1 \mathbf{x}^\top$, $\frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \delta_1$ |
| Gradient descent | $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$ |

---

## Activation Functions

| Name | Formula | Derivative | Range |
|---|---|---|---|
| Sigmoid | $\frac{1}{1 + e^{-z}}$ | $a(1 - a)$ | $(0, 1)$ |
| Tanh | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - a^2$ | $(-1, 1)$ |
| ReLU | $\max(0, z)$ | $1_{z > 0}$ | $[0, \infty)$ |
| Leaky ReLU | $\max(\alpha z, z)$ | $1_{z > 0} + \alpha 1_{z \leq 0}$ | $(-\infty, \infty)$ |
| ELU | $z$ if $z > 0$, $\alpha(e^z - 1)$ else | $1_{z > 0} + \text{ELU}(z) + \alpha$ | $(-\alpha, \infty)$ |
| Swish | $z \cdot \sigma(\beta z)$ | $\sigma(\beta z) + \beta z \sigma(\beta z)(1 - \sigma(\beta z))$ | $(-\infty, \infty)$ |
| GELU | $z \cdot \Phi(z)$ | $\Phi(z) + z \cdot \phi(z)$ | $(-\infty, \infty)$ |

---

## Loss Functions

| Name | Formula | Derivative w.r.t. $\hat{y}$ | Task |
|---|---|---|---|
| MSE | $\frac{1}{m}\sum (y - \hat{y})^2$ | $-2(y - \hat{y})$ | Regression |
| BCE | $-\frac{1}{m}\sum [y\log\hat{y} + (1-y)\log(1-\hat{y})]$ | $-\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$ | Binary classification |
| CCE | $-\frac{1}{m}\sum\sum y_k \log \hat{y}_k$ | $\hat{y}_k - y_k$ (with softmax) | Multi-class |
| Hinge | $\frac{1}{m}\sum \max(0, 1 - y\hat{y})$ | $-y$ if $y\hat{y} < 1$ else $0$ | SVM / margin |
| Huber | Quadratic for $\|y - \hat{y}\| \leq \delta$, linear otherwise | $-(y - \hat{y})$ or $-\delta$ | Robust regression |

---

## Weight Initialization

| Method | Distribution | Variance | Use With |
|---|---|---|---|
| Random small | $\mathcal{N}(0, 0.0001)$ | $0.0001$ | Shallow nets |
| Xavier (Glorot) | $\mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$ | $\frac{2}{n_{\text{in}} + n_{\text{out}}}$ | Tanh, Sigmoid |
| He (Kaiming) | $\mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$ | $\frac{2}{n_{\text{in}}}$ | ReLU, Leaky ReLU |

---

## Common Shapes

| Variable | Shape | Meaning |
|---|---|---|
| $\mathbf{X}$ | $(m, n_{\text{in}})$ | Batch of $m$ samples |
| $\mathbf{W}^{[l]}$ | $(n_{\text{in}}^{[l]}, n_{\text{out}}^{[l]})$ | Weights for layer $l$ |
| $\mathbf{b}^{[l]}$ | $(1, n_{\text{out}}^{[l]})$ | Bias for layer $l$ |
| $\mathbf{a}^{[l]}$ | $(m, n_{\text{out}}^{[l]})$ | Activations of layer $l$ |
| $\mathbf{\delta}^{[l]}$ | $(m, n_{\text{out}}^{[l]})$ | Error at layer $l$ |

---

## Vanishing / Exploding Gradients — Solutions

| Problem | Solution | Mechanism |
|---|---|---|
| Vanishing | ReLU activation | Gradient $=1$ for $z > 0$ |
| Vanishing | He initialization | Preserves activation variance |
| Vanishing | Batch normalization | Keeps activations in linear regime |
| Vanishing | Residual connections | Identity path for gradient flow |
| Exploding | Gradient clipping | Normalizes gradient norm |
| Exploding | Weight regularization | Penalizes large weights |
