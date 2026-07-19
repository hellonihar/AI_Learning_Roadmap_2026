# Calculus — Exercises

## Conceptual Questions

1. **Chain Rule in Backpropagation**: A neural network computes $L = \frac{1}{2}(y - \sigma(wx + b))^2$ where $\sigma(z) = 1/(1+e^{-z})$. Write the expression for $\partial L / \partial w$ using the chain rule, identifying each intermediate variable.

2. **Gradient Properties**: The gradient $\nabla f(\mathbf{x})$ points in the direction of steepest ascent. Explain why gradient descent uses $-\nabla f(\mathbf{x})$ rather than moving along a random direction. What role does the learning rate play?

3. **Hessian Interpretation**: A function $f(x, y)$ has Hessian $\mathbf{H} = \begin{bmatrix} 2 & 0 \\ 0 & -2 \end{bmatrix}$ at a critical point. Is this point a minimum, maximum, or saddle point? How can you tell?

4. **Taylor Series and Gradient Descent**: Gradient descent can be derived from a first-order Taylor approximation. Show that minimizing $f(\mathbf{x} + \Delta \mathbf{x})$ approximated by $f(\mathbf{x}) + \nabla f(\mathbf{x})^T \Delta \mathbf{x}$ subject to $\|\Delta \mathbf{x}\| \leq \epsilon$ leads to the update $\Delta \mathbf{x} = -\eta \nabla f(\mathbf{x})$.

## Coding Exercises

5. **Numerical Gradient Checker**: Write a function `check_gradient(f, grad_f, x)` that compares the analytical gradient with a numerical approximation using central differences. Test it on:
   - $f(x) = \sin(x^2) + 3x$ with gradient $2x \cos(x^2) + 3$
   - $f(\mathbf{x}) = \mathbf{x}^T \mathbf{A} \mathbf{x}$ with gradient $(\mathbf{A} + \mathbf{A}^T)\mathbf{x}$

6. **Logistic Regression from Scratch**: Implement one step of gradient descent for logistic regression:
   - Forward: $\hat{y} = \sigma(\mathbf{X}\mathbf{w} + b)$
   - Loss: $L = -\frac{1}{n} \sum [y \log(\hat{y}) + (1-y)\log(1-\hat{y})]$
   - Compute gradients manually for $\mathbf{w}$ and $b$ using the chain rule
   - Verify with numerical gradients
   Use $n=100$, $d=5$ random data.

7. **Learning Rate Sensitivity**: Implement gradient descent for $f(x) = x^4 - 3x^3 + 2$. Try learning rates $\eta \in \{0.001, 0.01, 0.1, 1.0\}$. For each:
   - Run 100 iterations starting from $x=3$
   - Report the final $x$ and $f(x)$
   - Describe what happens (convergence, divergence, oscillation)

---

## Answers

**Answer 1**: Let $z = wx + b$, $a = \sigma(z) = 1/(1+e^{-z})$, $L = \frac{1}{2}(y-a)^2$. Then:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w} = -(y-a) \cdot a(1-a) \cdot x
$$

**Answer 2**: At any point, the directional derivative is maximized in the gradient direction. Gradient descent moves opposite to ascent (downhill). The learning rate controls step size: too large may overshoot/oscillate, too small converges slowly.

**Answer 3**: Saddle point. The eigenvalues of $\mathbf{H}$ are $2$ (positive) and $-2$ (negative). The function curves upward in $x$ and downward in $y$, indicating neither a min nor max.

**Answer 4**: Minimizing $f(\mathbf{x}) + \nabla f(\mathbf{x})^T \Delta \mathbf{x}$ with $\|\Delta \mathbf{x}\| \leq \epsilon$: the minimizer is $\Delta \mathbf{x} = -\epsilon \nabla f(\mathbf{x}) / \|\nabla f(\mathbf{x})\|$. Setting $\eta = \epsilon / \|\nabla f(\mathbf{x})\|$ gives the standard update.

**Answer 5**:
```python
import numpy as np

def check_gradient(f, grad_f, x, h=1e-7):
    x = np.asarray(x, dtype=float)
    numerical = np.zeros_like(x)
    for i in range(len(x)):
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[i] += h; x_minus[i] -= h
        numerical[i] = (f(x_plus) - f(x_minus)) / (2*h)
    analytical = grad_f(x)
    return np.max(np.abs(numerical - analytical))

f1 = lambda x: np.sin(x**2) + 3*x
g1 = lambda x: 2*x*np.cos(x**2) + 3
print(f"f1 diff: {check_gradient(f1, g1, 1.5):.2e}")

A = np.array([[2, 1], [1, 3]])
f2 = lambda x: x @ A @ x
g2 = lambda x: (A + A.T) @ x
print(f"f2 diff: {check_gradient(f2, g2, np.array([1.0, 2.0])):.2e}")
```

**Answer 6**:
```python
np.random.seed(42)
n, d = 100, 5
X = np.random.randn(n, d)
y = np.random.randint(0, 2, n)
w = np.zeros(d)
b = 0.0

z = X @ w + b
y_pred = 1 / (1 + np.exp(-z))
loss = -np.mean(y * np.log(y_pred + 1e-15) + (1-y) * np.log(1 - y_pred + 1e-15))
dl_dw = X.T @ (y_pred - y) / n
dl_db = np.mean(y_pred - y)

def numerical_grad(w, b, X, y):
    h = 1e-7
    gw = np.zeros_like(w)
    for i in range(len(w)):
        wp = w.copy(); wp[i] += h
        wm = w.copy(); wm[i] -= h
        zp = X @ wp + b; yp = 1/(1+np.exp(-zp))
        zm = X @ wm + b; ym = 1/(1+np.exp(-zm))
        Lp = -np.mean(y*np.log(yp+1e-15) + (1-y)*np.log(1-yp+1e-15))
        Lm = -np.mean(y*np.log(ym+1e-15) + (1-y)*np.log(1-ym+1e-15))
        gw[i] = (Lp - Lm) / (2*h)
    gb = (loss_fn(X, w, b+h, y) - loss_fn(X, w, b-h, y)) / (2*h)
    return gw, gb
```

**Answer 7**:
```python
def gd(f, grad, x0, lr, steps=100):
    x = x0
    path = [x]
    for _ in range(steps):
        x = x - lr * grad(x)
        path.append(x)
    return x, f(x)

f = lambda x: x**4 - 3*x**3 + 2
g = lambda x: 4*x**3 - 9*x**2

for lr in [0.001, 0.01, 0.1, 1.0]:
    x_final, f_final = gd(f, g, 3.0, lr)
    print(f"lr={lr:.3f}: x={x_final:.4f}, f(x)={f_final:.4f}")
```
Small lr converges slowly; large lr (1.0) may diverge due to steep gradients.
