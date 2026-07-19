# Calculus with NumPy

This walkthrough demonstrates calculus concepts numerically using only NumPy.

```python
import numpy as np
```

## 1. Numerical Derivatives

```python
def numerical_derivative(f, x, h=1e-7):
    """Central difference approximation of f'(x)"""
    return (f(x + h) - f(x - h)) / (2 * h)

# Test: f(x) = x^3, f'(x) = 3x^2
f = lambda x: x**3
for x in [0.5, 1.0, 2.0]:
    numerical = numerical_derivative(f, x)
    analytical = 3 * x**2
    print(f"f'({x:.1f}): numerical={numerical:.6f}, analytical={analytical:.6f}")

# Partial derivatives: f(x, y) = x^2 * y + y^3
def partial_x(f, x, y, h=1e-7):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

def partial_y(f, x, y, h=1e-7):
    return (f(x, y + h) - f(x, y - h)) / (2 * h)

f_mv = lambda x, y: x**2 * y + y**3
x, y = 2.0, 3.0
print(f"\n∂f/∂x at ({x},{y}): numerical={partial_x(f_mv, x, y):.4f}, analytical={2*x*y:.4f}")
print(f"∂f/∂y at ({x},{y}): numerical={partial_y(f_mv, x, y):.4f}, analytical={x**2 + 3*y**2:.4f}")
```

## 2. Chain Rule Verification

```python
# f(x) = sin(x^2 + 1), f'(x) = cos(x^2 + 1) * 2x
def f_composite(x):
    return np.sin(x**2 + 1)

def analytical_chain(x):
    inner = x**2 + 1
    return np.cos(inner) * 2 * x

x_vals = np.array([0.5, 1.0, 1.5])
numerical = numerical_derivative(f_composite, x_vals)
analytical = analytical_chain(x_vals)
print("Chain rule verification:")
for i, x in enumerate(x_vals):
    print(f"  x={x:.1f}: numerical={numerical[i]:.6f}, analytical={analytical[i]:.6f}")

# Multivariate chain rule: z = f(x, y), x = t^2, y = 2t
# dz/dt = ∂f/∂x * dx/dt + ∂f/∂y * dy/dt
def f_xy(x, y):
    return x**2 * y + y**2

def dz_dt_analytical(t):
    x, y = t**2, 2*t
    dx_dt, dy_dt = 2*t, 2
    # ∂f/∂x = 2xy, ∂f/∂y = x^2 + 2y
    return (2*x*y) * dx_dt + (x**2 + 2*y) * dy_dt

def dz_dt_numerical(t, h=1e-7):
    x = lambda t: t**2
    y = lambda t: 2*t
    return (f_xy(x(t+h), y(t+h)) - f_xy(x(t-h), y(t-h))) / (2*h)

for t in [0.5, 1.0, 2.0]:
    print(f"dz/dt at t={t:.1f}: num={dz_dt_numerical(t):.4f}, ana={dz_dt_analytical(t):.4f}")
```

## 3. Gradient Descent

```python
# Minimize f(x) = x^2 + 2x + 1 using gradient descent
# f'(x) = 2x + 2
def f_quad(x):
    return x**2 + 2*x + 1

def grad_f_quad(x):
    return 2*x + 2

x = 4.0  # Starting point
lr = 0.1
path = [x]

for i in range(20):
    x = x - lr * grad_f_quad(x)
    path.append(x)
    if i % 5 == 0:
        print(f"Step {i}: x={x:.6f}, f(x)={f_quad(x):.6f}")

print(f"\nFinal: x={x:.6f} (expected minimum at x=-1)")
print(f"f(x)={f_quad(x):.6f} (expected 0)")

# 2D Rosenbrock function: f(x, y) = (a-x)^2 + b(y-x^2)^2
def rosenbrock(xy, a=1, b=100):
    x, y = xy
    return (a - x)**2 + b * (y - x**2)**2

def grad_rosenbrock(xy, a=1, b=100):
    x, y = xy
    dx = -2*(a - x) - 4*b*x*(y - x**2)
    dy = 2*b*(y - x**2)
    return np.array([dx, dy])

xy = np.array([0.0, 0.0])
lr = 0.001
for i in range(1000):
    xy = xy - lr * grad_rosenbrock(xy)
    if i % 200 == 0:
        print(f"Step {i}: xy={xy}, f={rosenbrock(xy):.6f}")

print(f"\nRosenbrock minimum: xy={xy} (expected [1, 1])")
```

## 4. Gradient of a Simple Neural Network Layer

```python
# Compute gradient through a linear layer + sigmoid + MSE loss
np.random.seed(42)
n_features = 3
n_samples = 5

X = np.random.randn(n_samples, n_features)
w = np.random.randn(n_features)
b = np.random.randn(1)
y_true = np.random.randn(n_samples)

# Forward pass
z = X @ w + b  # linear transformation
y_pred = 1 / (1 + np.exp(-z))  # sigmoid activation
loss = np.mean((y_pred - y_true)**2)  # MSE
print(f"Initial loss: {loss:.6f}")

# Backward pass (manual gradients)
dl_dy_pred = 2 * (y_pred - y_true) / n_samples  # dMSE/dy_pred
dy_pred_dz = y_pred * (1 - y_pred)  # dsigmoid/dz
dz_dw = X  # dz/dw = X
dz_db = np.ones(n_samples)  # dz/db

# Chain rule
dl_dw = dz_dw.T @ (dl_dy_pred * dy_pred_dz)
dl_db = np.sum(dl_dy_pred * dy_pred_dz)

print(f"Gradient w.r.t. w: {dl_dw}")
print(f"Gradient w.r.t. b: {dl_db}")

# Verify with numerical gradient
def loss_fn(w, b):
    z = X @ w + b
    y_pred = 1 / (1 + np.exp(-z))
    return np.mean((y_pred - y_true)**2)

w_num_grad = np.zeros_like(w)
h = 1e-7
for i in range(len(w)):
    w_plus = w.copy(); w_plus[i] += h
    w_minus = w.copy(); w_minus[i] -= h
    w_num_grad[i] = (loss_fn(w_plus, b) - loss_fn(w_minus, b)) / (2*h)

b_num_grad = (loss_fn(w, b + h) - loss_fn(w, b - h)) / (2*h)

print(f"\nNumerical gradient w: {w_num_grad}")
print(f"Match: {np.allclose(dl_dw, w_num_grad, rtol=1e-4)}")
print(f"Numerical gradient b: {b_num_grad:.6f}")
print(f"Match: {np.allclose(dl_db, b_num_grad, rtol=1e-4)}")
```

## 5. Computing the Jacobian

```python
# Jacobian of a simple vector function f(x) = [x0^2 + x1, x0 * x1^2]
def f_vec(x):
    return np.array([x[0]**2 + x[1], x[0] * x[1]**2])

def jacobian_numerical(f, x, h=1e-7):
    m = len(f(x))
    n = len(x)
    J = np.zeros((m, n))
    for i in range(n):
        x_plus = x.copy(); x_plus[i] += h
        x_minus = x.copy(); x_minus[i] -= h
        J[:, i] = (f(x_plus) - f(x_minus)) / (2*h)
    return J

x = np.array([2.0, 3.0])
J = jacobian_numerical(f_vec, x)
print(f"Jacobian at x={x}:")
print(J)
print(f"Expected: [[{2*x[0]}, {1}], [{x[1]**2}, {2*x[0]*x[1]}]]")
```

## 6. Taylor Series Approximation

```python
# Approximate f(x) = e^x using Taylor series around a=0
def exp_taylor(x, n_terms):
    approx = 0
    for n in range(n_terms):
        approx += x**n / np.math.factorial(n)
    return approx

x = 0.5
true_val = np.exp(x)
print(f"True e^{x} = {true_val:.6f}")
for k in [1, 2, 4, 6]:
    approx = exp_taylor(x, k)
    print(f"  {k}-term Taylor: {approx:.6f} (error: {abs(true_val - approx):.2e})")

# Multivariate Taylor: linear approximation of f(x,y) = x^2 * y
# f(x+dx, y+dy) ≈ f(x,y) + ∇f · [dx, dy]
x0, y0 = 1.0, 2.0
f0 = x0**2 * y0
grad = np.array([2*x0*y0, x0**2])  # [4, 1]

dx, dy = 0.1, -0.05
true_val = (x0+dx)**2 * (y0+dy)
linear_approx = f0 + grad @ np.array([dx, dy])
print(f"\nf({x0+dx}, {y0+dy}): true={true_val:.6f}, linear approx={linear_approx:.6f}")
print(f"Error: {abs(true_val - linear_approx):.4f}")
```

These numerical techniques mirror what automatic differentiation does internally in frameworks like PyTorch and TensorFlow.
