# Calculus for Machine Learning

Calculus provides the mathematical framework for understanding how functions change, which is essential for optimizing machine learning models. Every training loop relies on gradients to update parameters.

---

## 1. Derivatives

The **derivative** of a function $f(x)$ measures its instantaneous rate of change:

$$
f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

### Basic Rules

- **Power rule**: $\frac{d}{dx} x^n = n x^{n-1}$
- **Product rule**: $(fg)' = f'g + fg'$
- **Quotient rule**: $(f/g)' = (f'g - fg') / g^2$
- **Chain rule**: $\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$

The chain rule is the cornerstone of backpropagation in neural networks.

### Partial Derivatives

For a multivariate function $f(x_1, x_2, \ldots, x_n)$, the partial derivative $\frac{\partial f}{\partial x_i}$ holds all other variables constant:

$$
\frac{\partial}{\partial x} (x^2 y + y^2) = 2xy
$$

### Total Derivative

When variables are interdependent, the **total derivative** accounts for indirect dependencies:

$$
\frac{df}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}
$$

---

## 2. The Chain Rule

The chain rule computes derivatives of composite functions. For $z = f(g(x))$:

$$
\frac{dz}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
$$

### Multivariate Chain Rule

For $z = f(x, y)$ where $x = g(t)$ and $y = h(t)$:

$$
\frac{dz}{dt} = \frac{\partial z}{\partial x} \frac{dx}{dt} + \frac{\partial z}{\partial y} \frac{dy}{dt}
$$

### Backpropagation

In a neural network with loss $L$, weight $w$, and activation $a = \sigma(z)$ where $z = wx + b$:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

This sequential application of the chain rule computes gradients from the output layer back to the input layer.

---

## 3. Gradients and Gradient Descent

The **gradient** of a scalar function $f: \mathbb{R}^n \to \mathbb{R}$ is a vector of all partial derivatives:

$$
\nabla f(\mathbf{x}) = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n} \right]^T
$$

The gradient points in the direction of **steepest ascent**. **Gradient descent** moves in the opposite direction to minimize a function:

$$
\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \eta \nabla f(\mathbf{x}^{(t)})
$$

where $\eta$ is the **learning rate**.

### Gradient of Common Loss Functions

**Mean Squared Error**: $L = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$

$$
\frac{\partial L}{\partial \hat{y}_i} = -\frac{2}{n}(y_i - \hat{y}_i)
$$

**Cross-Entropy Loss**: $L = -\sum_{i} y_i \log(\hat{y}_i)$

$$
\frac{\partial L}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i}
$$

---

## 4. Multivariate Calculus

### Jacobian Matrix

For a vector-valued function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian $\mathbf{J} \in \mathbb{R}^{m \times n}$ contains all first-order partial derivatives:

$$
\mathbf{J} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

The Jacobian generalizes the gradient to vector outputs. It is used in backpropagation through vector-valued layers.

### Hessian Matrix

The Hessian $\mathbf{H} \in \mathbb{R}^{n \times n}$ contains all second-order partial derivatives of a scalar function:

$$
\mathbf{H}_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

The Hessian characterizes the **curvature** of a function:
- Positive definite $\implies$ local minimum
- Negative definite $\implies$ local maximum
- Indefinite $\implies$ saddle point

Second-order optimization methods (e.g., Newton's method) use the Hessian, but are often too expensive for deep learning due to $O(n^2)$ memory cost.

---

## 5. Taylor Series

Taylor series approximates a function around a point using its derivatives:

$$
f(x) = f(a) + f'(a)(x - a) + \frac{f''(a)}{2!}(x - a)^2 + \frac{f'''(a)}{3!}(x - a)^3 + \ldots
$$

### Multivariate Taylor Expansion (to second order)

$$
f(\mathbf{x} + \Delta \mathbf{x}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^T \Delta \mathbf{x} + \frac{1}{2} \Delta \mathbf{x}^T \mathbf{H}(\mathbf{x}) \Delta \mathbf{x}
$$

The first-order term gives gradient descent. The second-order term (with the Hessian) improves the approximation and is used in Newton's method and natural gradient descent.

---

## 6. Applications in AI/ML

| Concept | Application |
|---------|------------|
| Partial derivatives | Computing gradients for each parameter in a model |
| Chain rule | Backpropagation through computational graphs |
| Gradient descent | Optimizing all neural network loss functions |
| Jacobian | Computing gradients of softmax, batchnorm layers |
| Hessian | Understanding optimization landscape, saddle points |
| Taylor series | Deriving gradient descent, analyzing convergence |

### Backpropagation Example

For a simple network with one hidden layer:

$$
\hat{y} = \sigma(\mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2)
$$

The chain rule gives:

$$
\frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial \mathbf{z}_2} \cdot \frac{\partial \mathbf{z}_2}{\partial \mathbf{h}} \cdot \frac{\partial \mathbf{h}}{\partial \mathbf{z}_1} \cdot \frac{\partial \mathbf{z}_1}{\partial \mathbf{W}_1}
$$

Each term is a Jacobian or gradient, and automatic differentiation frameworks (PyTorch, TensorFlow) compute these products efficiently.

---

## Summary

Calculus, especially differentiation, is the engine that drives learning in neural networks. Understanding gradients, the chain rule, and their extensions to multivariate functions gives you the tools to design, debug, and improve optimization algorithms.
