# 04 — Loss Functions

The loss function quantifies how far the network's predictions are from the true targets. Choosing the right loss is critical for gradient-based learning.

---

## 1. Mean Squared Error (MSE) — Regression

$$\mathcal{L}_{\text{MSE}} = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2$$

**Derivative w.r.t. prediction (per sample):**

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -2 (y - \hat{y})$$

| Pros | Cons |
|---|---|
| Simple, convex for linear models | Penalizes outliers quadratically (sensitive) |
| Smooth, always differentiable | Slow convergence with sigmoid/tanh due to small gradients when saturated |

**Use when:** Regression problems with Gaussian noise assumption.

---

## 2. Binary Cross-Entropy (BCE) — Binary Classification

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

**Derivative w.r.t. prediction (per sample):**

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}}$$

When combined with sigmoid output, the gradient simplifies to $\hat{y} - y$ (as shown in the backpropagation chapter).

| Pros | Cons |
|---|---|
| Strong gradients even when predictions are wrong | Numerically unstable if $\hat{y}=0$ or $\hat{y}=1$ |
| Probabilistic interpretation (MLE) | Requires clipping or log-sum-exp trick |

**Use when:** Binary classification (fraud detection, spam, etc.).

---

## 3. Categorical Cross-Entropy (CCE) — Multi-Class

$$\mathcal{L}_{\text{CCE}} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log \hat{y}_k^{(i)}$$

where $K$ is the number of classes and $\mathbf{y}$ is one-hot encoded. The output layer uses **softmax**:

$$\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

**Gradient with softmax + CCE:**

$$\frac{\partial \mathcal{L}}{\partial z_k} = \hat{y}_k - y_k$$

**Use when:** Multi-class classification (image recognition, document classification).

---

## 4. Hinge Loss — SVM / Max-Margin

$$\mathcal{L}_{\text{hinge}} = \frac{1}{m} \sum_{i=1}^{m} \max\left(0, 1 - y^{(i)} \cdot \hat{y}^{(i)}\right)$$

where $y \in \{-1, +1\}$ and $\hat{y}$ is the raw output (before activation).

**Derivative w.r.t. prediction (per sample):**

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = \begin{cases} -y & \text{if } y \cdot \hat{y} < 1 \\ 0 & \text{otherwise} \end{cases}$$

| Pros | Cons |
|---|---|
| Encourages margin maximization | Not differentiable at $y\cdot\hat{y} = 1$ |
| More robust to outliers than cross-entropy | Rarely used in modern deep nets |

**Use when:** SVMs, or when max-margin behavior is desired.

---

## 5. Huber Loss — Robust Regression

$$\mathcal{L}_{\text{Huber}} = \frac{1}{m} \sum_{i=1}^{m} \begin{cases} \frac{1}{2} (y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta |y - \hat{y}| - \frac{1}{2} \delta^2 & \text{otherwise} \end{cases}$$

**Derivative w.r.t. prediction (per sample):**

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = \begin{cases} -(y - \hat{y}) & \text{if } |y - \hat{y}| \leq \delta \\ -\delta \cdot \text{sign}(y - \hat{y}) & \text{otherwise} \end{cases}$$

| Pros | Cons |
|---|---|
| Quadratic for small errors (like MSE) | One extra hyperparameter $\delta$ to tune |
| Linear for large errors (robust to outliers like MAE) | Less common than MSE |
| Differentiable everywhere | |

**Use when:** Regression with outliers.

---

## Quick Reference

| Loss | Task | Output Activation | Gradient Form |
|---|---|---|---|
| MSE | Regression | Linear (none) | $-2(y - \hat{y})$ |
| BCE | Binary classification | Sigmoid | $\hat{y} - y$ |
| CCE | Multi-class classification | Softmax | $\hat{y}_k - y_k$ |
| Hinge | Binary (margin) | Linear or Tanh | $-y$ if $y\hat{y} < 1$ |
| Huber | Robust regression | Linear | $-(y - \hat{y})$ or $-\delta$ |

---

**Next:** [05 - Weight Initialization.md](./05%20-%20Weight%20Initialization.md)
