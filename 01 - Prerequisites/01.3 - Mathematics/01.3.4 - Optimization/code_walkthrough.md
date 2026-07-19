# Optimization with NumPy

This walkthrough implements various optimization algorithms using only NumPy.

```python
import numpy as np
```

## 1. Convex vs. Non-Convex Functions

```python
# Convex: f(x) = x^2 (Hessian = 2 > 0 everywhere)
# Non-convex: f(x) = x^4 - 10x^2 (has two local minima, one local max)
def convex_fn(x):
    return x**2

def nonconvex_fn(x):
    return x**4 - 10*x**2

# Visualize (via values)
x_vals = np.linspace(-4, 4, 100)
convex_vals = convex_fn(x_vals)
nonconvex_vals = nonconvex_fn(x_vals)
print(f"Convex min at x=0: f(0)={convex_fn(0)}")
print(f"Non-convex minima: f({np.sqrt(5):.3f})={nonconvex_fn(np.sqrt(5)):.3f}, "
      f"f({-np.sqrt(5):.3f})={nonconvex_fn(-np.sqrt(5)):.3f}")

# A 2D convex function: f(x,y) = x^2 + y^2 (bowl shape)
# A 2D non-convex function: f(x,y) = x^2 - y^2 (saddle)
def saddle_fn(xy):
    x, y = xy
    return x**2 - y**2

# Hessian of saddle: [[2, 0], [0, -2]] — indefinite!
```

## 2. Gradient Descent Implementations

```python
np.random.seed(42)

# Objective: f(w) = ||Xw - y||^2 (linear regression)
n, d = 100, 5
X = np.random.randn(n, d)
w_true = np.random.randn(d)
y = X @ w_true + 0.1 * np.random.randn(n)

def loss(w, X, y):
    return np.mean((X @ w - y)**2)

def grad_loss(w, X, y):
    return 2 * X.T @ (X @ w - y) / len(y)

# Full Batch Gradient Descent
def batch_gd(X, y, lr=0.1, n_iter=100):
    w = np.zeros(X.shape[1])
    loss_history = []
    for i in range(n_iter):
        w = w - lr * grad_loss(w, X, y)
        loss_history.append(loss(w, X, y))
    return w, loss_history

# Stochastic Gradient Descent
def sgd(X, y, lr=0.1, n_epochs=10):
    w = np.zeros(X.shape[1])
    n = len(y)
    loss_history = []
    for epoch in range(n_epochs):
        idx = np.random.permutation(n)
        X_shuffled, y_shuffled = X[idx], y[idx]
        for i in range(n):
            xi = X_shuffled[i:i+1]
            yi = y_shuffled[i:i+1]
            grad = 2 * xi.T @ (xi @ w - yi)
            w = w - lr * grad
        loss_history.append(loss(w, X, y))
    return w, loss_history

# Mini-Batch SGD
def minibatch_sgd(X, y, lr=0.1, n_epochs=10, batch_size=16):
    w = np.zeros(X.shape[1])
    n = len(y)
    loss_history = []
    for epoch in range(n_epochs):
        idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            Xb, yb = X[batch_idx], y[batch_idx]
            grad = 2 * Xb.T @ (Xb @ w - yb) / len(batch_idx)
            w = w - lr * grad
        loss_history.append(loss(w, X, y))
    return w, loss_history

w_batch, hist_batch = batch_gd(X, y)
w_sgd, hist_sgd = sgd(X, y)
w_mb, hist_mb = minibatch_sgd(X, y)

print(f"Final loss — Batch GD: {hist_batch[-1]:.6f}")
print(f"Final loss — SGD: {hist_sgd[-1]:.6f}")
print(f"Final loss — Mini-batch: {hist_mb[-1]:.6f}")
print(f"w_true: {w_true.round(3)}")
print(f"w_batch: {w_batch.round(3)}")
```

## 3. Momentum

```python
def sgd_momentum(X, y, lr=0.1, beta=0.9, n_epochs=10, batch_size=16):
    w = np.zeros(X.shape[1])
    v = np.zeros(X.shape[1])
    n = len(y)
    loss_history = []
    for epoch in range(n_epochs):
        idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            Xb, yb = X[batch_idx], y[batch_idx]
            grad = 2 * Xb.T @ (Xb @ w - yb) / len(batch_idx)
            v = beta * v + grad
            w = w - lr * v
        loss_history.append(loss(w, X, y))
    return w, loss_history

w_mom, hist_mom = sgd_momentum(X, y)
print(f"Momentum final loss: {hist_mom[-1]:.6f}")

# Compare convergence speed
print(f"\nLoss after 1 epoch:")
print(f"  Batch GD: {hist_batch[0]:.6f}")
print(f"  Momentum: {hist_mom[0]:.6f}")
print(f"  Mini-batch: {hist_mb[0]:.6f}")
```

## 4. Adam Optimizer

```python
def adam(X, y, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, n_epochs=10, batch_size=32):
    w = np.zeros(X.shape[1])
    m = np.zeros(X.shape[1])
    v = np.zeros(X.shape[1])
    n = len(y)
    loss_history = []
    t = 0
    for epoch in range(n_epochs):
        idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            t += 1
            batch_idx = idx[i:i+batch_size]
            Xb, yb = X[batch_idx], y[batch_idx]
            grad = 2 * Xb.T @ (Xb @ w - yb) / len(batch_idx)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        loss_history.append(loss(w, X, y))
    return w, loss_history

w_adam, hist_adam = adam(X, y)
print(f"Adam final loss: {hist_adam[-1]:.6f}")

# Compare all methods
methods = [
    ("Batch GD", hist_batch),
    ("SGD", hist_sgd),
    ("Mini-batch", hist_mb),
    ("Momentum", hist_mom),
    ("Adam", hist_adam),
]
print("\nConvergence comparison (loss after epoch 1 vs final):")
for name, hist in methods:
    print(f"  {name:12s}: epoch0={hist[0]:.4f}  final={hist[-1]:.6f}")
```

## 5. Lagrange Multipliers (Constrained Optimization)

```python
# Minimize f(x, y) = x^2 + y^2 subject to x + y = 1
# Lagrangian: L(x, y, λ) = x^2 + y^2 + λ(x + y - 1)
# ∂L/∂x = 2x + λ = 0 → x = -λ/2
# ∂L/∂y = 2y + λ = 0 → y = -λ/2
# ∂L/∂λ = x + y - 1 = 0 → -λ = 1 → λ = -1
# Solution: x = 0.5, y = 0.5

def solve_lagrange_quadratic():
    # Analytical solution
    A = np.array([[2, 0, 1],
                  [0, 2, 1],
                  [1, 1, 0]])
    b = np.array([0, 0, 1])
    sol = np.linalg.solve(A, b)
    x, y, lam = sol
    print(f"Lagrange solution: x={x:.4f}, y={y:.4f}, λ={lam:.4f}")
    print(f"f(x,y) = {x**2 + y**2:.4f}")
    print(f"Constraint: x + y = {x+y:.4f}")

solve_lagrange_quadratic()

# Numerical: projected gradient descent
def projected_gradient_descent(lr=0.1, n_iter=100):
    # Minimize f(x,y) = x^2 + y^2 s.t. x + y = 1
    # Step 1: gradient step on f
    # Step 2: project onto constraint x + y = 1
    xy = np.array([0.0, 0.0])
    for i in range(n_iter):
        grad = np.array([2*xy[0], 2*xy[1]])
        xy = xy - lr * grad
        # Project: find closest point on line x + y = 1
        # Min ||xy - z||^2 s.t. z0 + z1 = 1
        # z = xy + ((1 - xy[0] - xy[1]) / 2) * [1, 1]
        t = (1 - xy[0] - xy[1]) / 2
        xy = xy + np.array([t, t])
        if i % 25 == 0:
            print(f"  Step {i}: xy={xy}, f={xy[0]**2+xy[1]**2:.4f}")
    return xy

print("\nProjected gradient descent:")
proj_result = projected_gradient_descent()
print(f"Final: {proj_result}")
```

## 6. Regularization

```python
# Compare L1 and L2 regularization
def ridge_regression(X, y, lam=1.0):
    # w = (X^T X + λI)^{-1} X^T y
    n, d = X.shape
    I = np.eye(d)
    return np.linalg.solve(X.T @ X + lam * I, X.T @ y)

def lasso_coordinate_descent(X, y, lam=1.0, n_iter=100, tol=1e-4):
    """Coordinate descent for Lasso"""
    n, d = X.shape
    w = np.zeros(d)
    for iteration in range(n_iter):
        w_old = w.copy()
        for j in range(d):
            residual = y - X @ w + X[:, j] * w[j]
            rho = X[:, j] @ residual
            if rho < -lam/2:
                w[j] = (rho + lam/2) / (X[:, j] @ X[:, j])
            elif rho > lam/2:
                w[j] = (rho - lam/2) / (X[:, j] @ X[:, j])
            else:
                w[j] = 0
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w

# Generate data with sparsity
np.random.seed(42)
n, d = 50, 20
X = np.random.randn(n, d)
w_true_sparse = np.zeros(d)
w_true_sparse[:3] = [2.0, -1.5, 1.0]
y = X @ w_true_sparse + 0.1 * np.random.randn(n)

w_ridge = ridge_regression(X, y, lam=0.5)
w_lasso = lasso_coordinate_descent(X, y, lam=0.5)

print("True weights (sparse):")
print(w_true_sparse.round(2))
print("\nRidge (L2) — all non-zero:")
print(w_ridge.round(2))
print(f"  Non-zero: {np.sum(np.abs(w_ridge) > 0.01)}/{d}")
print("\nLasso (L1) — sparse:")
print(w_lasso.round(2))
print(f"  Non-zero: {np.sum(np.abs(w_lasso) > 0.01)}/{d}")
```

These implementations demonstrate the core optimization algorithms used throughout machine learning.
