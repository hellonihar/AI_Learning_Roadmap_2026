# Optimization — Exercises

## Conceptual Questions

1. **Convexity Check**: Is the function $f(x) = \log(1 + e^x)$ convex? Prove your answer by computing the second derivative. Why might this function be useful in ML?

2. **SGD vs. Batch GD**: You are training a linear regression model on 1 million data points with 100 features. Batch GD takes 10 seconds per epoch. SGD takes 0.01 seconds per sample. After 1 minute of training, which approach has likely made more progress? Explain tradeoffs.

3. **Adam Bias Correction**: Why does Adam include bias correction terms $1 - \beta_1^t$ and $1 - \beta_2^t$? What would happen to the early updates without them?

4. **Lagrange Multiplier Intuition**: The SVM objective $\min \frac{1}{2}\|\mathbf{w}\|^2$ subject to $y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1$ forces a margin of at least 1. Explain geometrically why maximizing the margin corresponds to minimizing $\|\mathbf{w}\|^2$.

## Coding Exercises

5. **Optimizer Comparison on Rosenbrock**: Implement and compare the convergence of Batch GD, GD with momentum, and Adam on the Rosenbrock function $f(x, y) = (1-x)^2 + 100(y - x^2)^2$ starting from $(-1.5, 1.0)$. Run each for 2000 iterations and report the final value and distance to the minimum at $(1, 1)$.

6. **Learning Rate Schedules**: Implement gradient descent with three learning rate schedules for $f(x) = x^4 + 2x^3$:
   - Constant: $\eta = 0.01$
   - Step decay: $\eta = 0.01 \times 0.5^{\lfloor t/50 \rfloor}$
   - Exponential decay: $\eta = 0.01 \times e^{-0.02 t}$
   - Cosine annealing: $\eta = 0.01 \times \frac{1}{2}(1 + \cos(\pi t / T))$
   Run for 200 iterations from $x=2$ and compare convergence.

7. **Regularization Path**: Implement L2-regularized logistic regression and trace the weight path as $\lambda$ varies from $10^{-3}$ to $10^3$ (log scale, 20 values). Use synthetic 2D data where the optimal decision boundary is moderately complex. Plot how the weights and test accuracy change with $\lambda$.

---

## Answers

**Answer 1**: $f'(x) = \frac{e^x}{1+e^x} = \sigma(x)$, $f''(x) = \sigma(x)(1-\sigma(x)) \geq 0$. Since $f''(x) \geq 0$ everywhere, $f$ is convex. This is the softplus function, a smooth approximation to ReLU and used in output units.

**Answer 2**: In 1 minute, Batch GD completes 6 epochs (60/10). SGD completes 6000 samples (60/0.01) — only 0.6% of one epoch. However, SGD makes far more updates (6000 vs 6). For large datasets, SGD typically makes more progress due to frequent updates, though each update is noisier.

**Answer 3**: At early steps, $m^{(t)}$ and $v^{(t)}$ are initialized to zero and biased toward zero. Without bias correction, early updates would be too small. Bias correction scales them up appropriately. After many steps, $(1-\beta^t) \to 1$ and correction becomes negligible.

**Answer 4**: The margin (distance between the decision boundary and the nearest points) is $2/\|\mathbf{w}\|$. Maximizing $2/\|\mathbf{w}\|$ is equivalent to minimizing $\|\mathbf{w}\|^2$. A larger margin provides better generalization.

**Answer 5**:
```python
import numpy as np

def rosenbrock(xy):
    x, y = xy
    return (1-x)**2 + 100*(y - x**2)**2

def grad_rosenbrock(xy):
    x, y = xy
    dx = -2*(1-x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

def optimizer_comparison():
    x0 = np.array([-1.5, 1.0])
    lr, n_iter = 0.001, 2000

    # Batch GD
    w = x0.copy()
    gd_path = [rosenbrock(w)]
    for _ in range(n_iter):
        w = w - lr * grad_rosenbrock(w)
        gd_path.append(rosenbrock(w))

    # Momentum
    w, v = x0.copy(), np.zeros(2)
    mom_path = [rosenbrock(w)]
    for _ in range(n_iter):
        g = grad_rosenbrock(w)
        v = 0.9*v + g
        w = w - lr * v
        mom_path.append(rosenbrock(w))

    # Adam
    w, m, v_ad = x0.copy(), np.zeros(2), np.zeros(2)
    adam_path = [rosenbrock(w)]
    for t in range(1, n_iter+1):
        g = grad_rosenbrock(w)
        m = 0.9*m + 0.1*g
        v_ad = 0.999*v_ad + 0.001*g**2
        m_h = m/(1-0.9**t); v_h = v_ad/(1-0.999**t)
        w = w - 0.01 * m_h / (np.sqrt(v_h) + 1e-8)
        adam_path.append(rosenbrock(w))

    print(f"GD final: {gd_path[-1]:.4e}, dist to (1,1): unknown")
    print(f"Momentum final: {mom_path[-1]:.4e}")
    print(f"Adam final: {adam_path[-1]:.4e}")

optimizer_comparison()
```

**Answer 6**:
```python
def gd_with_schedule(f, grad, x0, schedule_fn, n_iter=200):
    x = x0
    for t in range(n_iter):
        lr = schedule_fn(t)
        x = x - lr * grad(x)
    return x, f(x)

f = lambda x: x**4 + 2*x**3
g = lambda x: 4*x**3 + 6*x**2
schedules = {
    'constant': lambda t: 0.01,
    'step': lambda t: 0.01 * 0.5**(t//50),
    'exp': lambda t: 0.01 * np.exp(-0.02*t),
    'cosine': lambda t: 0.01 * 0.5 * (1 + np.cos(np.pi * t / 200)),
}
for name, sched in schedules.items():
    x_f, f_f = gd_with_schedule(f, g, 2.0, sched)
    print(f"{name:10s}: x={x_f:.4f}, f(x)={f_f:.4f}")
```

**Answer 7**: Implement logistic regression with L2 regularization: the loss becomes $L(\mathbf{w}) = \text{cross-entropy} + \frac{\lambda}{2}\|\mathbf{w}\|^2$. As $\lambda$ increases, weights shrink toward zero (underfitting); as $\lambda \to 0$, the model approaches unregularized logistic regression (potential overfitting). The optimal $\lambda$ balances bias and variance.
