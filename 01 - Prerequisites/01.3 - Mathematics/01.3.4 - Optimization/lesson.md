# Optimization for Machine Learning

Optimization is the engine that trains machine learning models. Every neural network, linear regression, and support vector machine relies on optimization algorithms to minimize a loss function by adjusting model parameters.

---

## 1. Convex vs. Non-Convex Optimization

### Convex Functions

A function $f: \mathbb{R}^n \to \mathbb{R}$ is **convex** if for any $\mathbf{x}, \mathbf{y}$ and $\lambda \in [0, 1]$:

$$
f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y})
$$

Equivalently, the Hessian $\nabla^2 f(\mathbf{x})$ is positive semidefinite for all $\mathbf{x}$.

**Properties of convex optimization:**
- Any local minimum is a global minimum.
- Convergence is well-understood and guaranteed.
- Examples: linear regression, logistic regression, linear SVMs.

### Non-Convex Functions

Deep neural networks have non-convex loss landscapes with many local minima and saddle points. Non-convex optimization:
- Has no guarantee of finding the global minimum.
- Relies on heuristics and careful hyperparameter tuning.
- Surprisingly, SGD often finds solutions that generalize well.

---

## 2. Gradient Descent and Its Variants

### Vanilla Gradient Descent

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla L(\mathbf{w}^{(t)})
$$

- $\eta$: learning rate
- Computes gradient over the entire dataset per step

**Batch GD** is deterministic but can be extremely slow for large datasets.

### Stochastic Gradient Descent (SGD)

Updates parameters using a single randomly chosen sample per step:

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla L_i(\mathbf{w}^{(t)})
$$

- Much faster per iteration
- Noisy updates can help escape shallow local minima
- Requires learning rate scheduling

### Mini-Batch SGD

Compromise between batch and stochastic:

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \nabla L_i(\mathbf{w}^{(t)})
$$

- Batch size $\mathcal{B}$ is typically 32, 64, 128, or 256
- Vectorizes well on GPU
- Reduces variance compared to pure SGD

### Momentum

Accelerates convergence by accumulating a velocity vector:

$$
\begin{aligned}
\mathbf{v}^{(t+1)} &= \beta \mathbf{v}^{(t)} + \nabla L(\mathbf{w}^{(t)}) \\
\mathbf{w}^{(t+1)} &= \mathbf{w}^{(t)} - \eta \mathbf{v}^{(t+1)}
\end{aligned}
$$

- $\beta$ is typically 0.9
- Smooths oscillations and accelerates through ravine-like regions

### Nesterov Accelerated Gradient (NAG)

A "look-ahead" variant of momentum:

$$
\begin{aligned}
\mathbf{v}^{(t+1)} &= \beta \mathbf{v}^{(t)} + \nabla L(\mathbf{w}^{(t)} - \beta \mathbf{v}^{(t)}) \\
\mathbf{w}^{(t+1)} &= \mathbf{w}^{(t)} - \eta \mathbf{v}^{(t+1)}
\end{aligned}
$$

NAG computes the gradient at the approximate future position, providing better convergence guarantees.

### AdaGrad

Adapts the learning rate per parameter based on historical gradients:

$$
\mathbf{w}^{(t+1)}_i = \mathbf{w}^{(t)}_i - \frac{\eta}{\sqrt{G^{(t)}_{ii} + \epsilon}} \nabla L_i(\mathbf{w}^{(t)})
$$

- $G^{(t)}_{ii} = \sum_{\tau=1}^t (\nabla L_i^{(\tau)})^2$
- Good for sparse features
- Learning rate monotonically decreases and can become too small

### RMSProp

Modifies AdaGrad with an exponentially decaying average of squared gradients:

$$
\begin{aligned}
\mathbb{E}[g^2]^{(t)} &= \gamma \mathbb{E}[g^2]^{(t-1)} + (1-\gamma) (\nabla L^{(t)})^2 \\
\mathbf{w}^{(t+1)} &= \mathbf{w}^{(t)} - \frac{\eta}{\sqrt{\mathbb{E}[g^2]^{(t)} + \epsilon}} \nabla L^{(t)}
\end{aligned}
$$

### Adam (Adaptive Moment Estimation)

Combines momentum and RMSProp — currently the most popular optimizer for deep learning:

$$
\begin{aligned}
m^{(t)} &= \beta_1 m^{(t-1)} + (1-\beta_1) \nabla L^{(t)} \\
v^{(t)} &= \beta_2 v^{(t-1)} + (1-\beta_2) (\nabla L^{(t)})^2 \\
\hat{m}^{(t)} &= \frac{m^{(t)}}{1-\beta_1^t}, \quad \hat{v}^{(t)} = \frac{v^{(t)}}{1-\beta_2^t} \\
\mathbf{w}^{(t+1)} &= \mathbf{w}^{(t)} - \eta \frac{\hat{m}^{(t)}}{\sqrt{\hat{v}^{(t)}} + \epsilon}
\end{aligned}
$$

- $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$
- Bias correction for first and second moment estimates
- Default optimizer for most deep learning tasks

---

## 3. Constrained Optimization and Lagrange Multipliers

To optimize $f(\mathbf{x})$ subject to $g(\mathbf{x}) = 0$, introduce the Lagrangian:

$$
\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda g(\mathbf{x})
$$

The solution satisfies the KKT conditions:

$$
\nabla f(\mathbf{x}^*) + \lambda \nabla g(\mathbf{x}^*) = 0, \quad g(\mathbf{x}^*) = 0
$$

### Inequality Constraints

For $f(\mathbf{x})$ subject to $g(\mathbf{x}) \leq 0$:

$$
\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda g(\mathbf{x}), \quad \lambda \geq 0
$$

Complementary slackness: $\lambda g(\mathbf{x}) = 0$ — either the constraint is active ($g = 0$) or the multiplier is zero ($\lambda = 0$).

### Application: Support Vector Machines

SVMs solve a constrained optimization problem to maximize the margin subject to correct classification:

$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1
$$

The Lagrangian leads to the dual formulation where only support vectors (with $\lambda_i > 0$) matter.

---

## 4. Regularization

Regularization constrains the model to prevent overfitting.

### L2 Regularization (Ridge / Weight Decay)

Adds a penalty proportional to the squared norm of weights:

$$
\tilde{L}(\mathbf{w}) = L(\mathbf{w}) + \frac{\lambda}{2} \|\mathbf{w}\|_2^2
$$

Gradient update becomes:

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta (\nabla L(\mathbf{w}^{(t)}) + \lambda \mathbf{w}^{(t)}) = (1 - \eta \lambda) \mathbf{w}^{(t)} - \eta \nabla L(\mathbf{w}^{(t)})
$$

The $1 - \eta \lambda$ term is the "weight decay" factor.

### L1 Regularization (Lasso)

Adds a penalty proportional to the absolute sum of weights:

$$
\tilde{L}(\mathbf{w}) = L(\mathbf{w}) + \lambda \|\mathbf{w}\|_1
$$

L1 encourages **sparsity** — many weights become exactly zero.

### Elastic Net

Combines L1 and L2:

$$
\tilde{L}(\mathbf{w}) = L(\mathbf{w}) + \lambda_1 \|\mathbf{w}\|_1 + \frac{\lambda_2}{2} \|\mathbf{w}\|_2^2
$$

---

## 5. Applications in AI/ML

| Concept | Application |
|---------|------------|
| Gradient descent | Training all neural networks |
| SGD | Large-scale learning (ImageNet, GPT) |
| Adam | Default optimizer for Transformers, CNNs |
| Momentum | Accelerating convergence in deep networks |
| Lagrange multipliers | SVMs, constrained policy optimization in RL |
| L1 regularization | Feature selection in high-dimensional data |
| L2 regularization | Weight decay in neural networks |

---

## Summary

Optimization transforms the abstract goal of "learning from data" into concrete parameter updates. Understanding the strengths and weaknesses of different optimizers — from vanilla GD to Adam — helps you train models faster and more reliably.
