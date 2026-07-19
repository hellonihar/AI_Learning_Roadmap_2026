# Exercises — Foundations of Neural Networks

---

## Conceptual Questions

### Q1: The XOR Problem

Explain why a single-layer Perceptron cannot learn the XOR function. Draw the truth table and show that the data is not linearly separable. How does adding one hidden layer solve this?

**Answer:**
The XOR truth table:

| x1 | x2 | y |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Plotting these four points on a 2D plane shows that no single straight line can separate the red (y=1) from the blue (y=0) points. A hidden layer transforms the input space — the hidden neurons learn new features that make the problem linearly separable in the final layer. With a hidden layer of 2+ neurons and a non-linear activation (e.g., tanh or ReLU), an MLP can represent the XOR function exactly.

---

### Q2: Vanishing Gradients with Sigmoid

Consider a 10-layer network with sigmoid activations. Each layer's activation derivative is at most $0.25$. Estimate the maximum possible gradient magnitude that reaches the first layer, relative to the gradient at the output layer.

**Answer:**
The gradient at layer $l$ involves a product of $L - l$ Jacobians, each with a factor $\sigma'(z) \leq 0.25$. For a 10-layer network ($L=10$), the gradient at layer $1$ is multiplied by at most $0.25^{9} \approx 3.8 \times 10^{-6}$ of the output gradient. In practice, with saturation, the product can be even smaller. This is why ReLU (gradient $1$ for $z>0$) is preferred.

---

### Q3: Universal Approximation Theorem — Limits

The Universal Approximation Theorem states that a single hidden layer can approximate any continuous function. Why do we still use deep networks?

**Answer:**
The theorem requires an **exponentially large** number of hidden neurons for some functions (the "curse of dimensionality"). Deep networks can represent the same function **exponentially more efficiently** by composing simple features hierarchically. Additionally, deep networks generalize better and are easier to optimize in practice, though the theorem does not guarantee learnability from finite data.

---

### Q4: Compare MSE and Cross-Entropy

For a binary classification problem with sigmoid output, why is Binary Cross-Entropy preferred over Mean Squared Error?

**Answer:**
The gradient of MSE with sigmoid is $\partial \mathcal{L}_{\text{MSE}} / \partial z = -2(y - \hat{y}) \cdot \hat{y}(1 - \hat{y})$. When the prediction is very wrong (e.g., $y=1$, $\hat{y} \approx 0$), the factor $\hat{y}(1 - \hat{y})$ is near $0$, causing a very small gradient. With BCE, $\partial \mathcal{L}_{\text{BCE}} / \partial z = \hat{y} - y$, which gives a large gradient proportional to the error. BCE is the maximum likelihood loss for Bernoulli outputs and leads to faster, more stable convergence.

---

## Coding Exercises

### Q5: Implement Backpropagation for a Missing Layer

The following code computes gradients for the output layer. Implement the missing backpropagation for the hidden layer.

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

# Forward pass
X = np.random.randn(10, 4)     # 10 samples, 4 features
W1 = np.random.randn(4, 8)     # hidden: 4 -> 8
b1 = np.zeros((1, 8))
W2 = np.random.randn(8, 1)     # output: 8 -> 1
b2 = np.zeros((1, 1))
y = np.random.randint(0, 2, (10, 1))

z1 = np.dot(X, W1) + b1
a1 = relu(z1)
z2 = np.dot(a1, W2) + b2
y_pred = sigmoid(z2)

# Output layer gradients
dz2 = y_pred - y
dW2 = np.dot(a1.T, dz2) / X.shape[0]
db2 = np.sum(dz2, axis=0, keepdims=True) / X.shape[0]

# TODO: Compute dW1 and db1
```

**Answer:**
```python
da1 = np.dot(dz2, W2.T)
dz1 = da1 * (z1 > 0).astype(float)
dW1 = np.dot(X.T, dz1) / X.shape[0]
db1 = np.sum(dz1, axis=0, keepdims=True) / X.shape[0]
```

---

### Q6: Experiment with Different Activations

Modify the NeuralNetwork class from `code_walkthrough.md` to replace ReLU with ELU in the hidden layer. Compare the training loss curves.

**Answer:**
```python
def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))

def elu_derivative(z, alpha=1.0):
    return np.where(z > 0, 1.0, alpha * np.exp(z))

# In forward: self.a1 = elu(self.z1)
# In backward: dz1 = da1 * elu_derivative(self.z1)
```

ELU typically converges slightly faster than ReLU on this task due to smoother gradients and negative values pushing mean activation closer to zero. However, it is computationally slower.

---

### Q7: Implement Gradient Clipping

Add gradient clipping to the training loop. Clip the **global norm** of all gradients to a maximum of $1.0$.

**Answer:**
```python
def clip_gradients(grads, max_norm=1.0):
    total_norm = 0.0
    for g in grads.values():
        total_norm += np.sum(g ** 2)
    total_norm = np.sqrt(total_norm)
    
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for k in grads:
            grads[k] *= scale
    return grads

# In training loop:
grads = nn.backward(y, y_pred)
grads = clip_gradients(grads, max_norm=1.0)
nn.update(grads, lr)
```

---

### Q8: Implement He Initialization Manually

Without using `np.sqrt`, implement He initialization for a weight matrix of shape $(100, 64)$ using NumPy's `randn` and verify the mean and variance of the activations after a forward pass through a ReLU layer.

**Answer:**
```python
n_in = 100
n_out = 64
W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
b = np.zeros((1, n_out))

X = np.random.randn(1000, n_in)  # 1000 samples
z = np.dot(X, W) + b
a = np.maximum(0, z)

print(f"Mean of activations: {np.mean(a):.4f}")
print(f"Variance of activations: {np.var(a):.4f}")
# Expected: variance ~ 1.0 (approximately)
```

With He initialization, the variance of activations after ReLU should be close to $1.0$, ensuring stable signal propagation through the network. With random small init ($\sigma=0.01$), the variance would be ~$0.0001$, causing vanishing signals.
