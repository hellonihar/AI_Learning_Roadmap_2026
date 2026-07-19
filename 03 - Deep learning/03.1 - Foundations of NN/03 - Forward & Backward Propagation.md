# 03 — Forward & Backward Propagation

This chapter walks through the full forward and backward pass for a **2-layer neural network** (one hidden layer, one output layer). We use sigmoid activation in the hidden layer and a single sigmoid output for binary classification.

---

## Network Architecture

```
Input:  x = [x₁, x₂, ..., xₙ]ᵗ
Hidden: z₁ = W₁ x + b₁,    a₁ = σ(z₁)
Output: z₂ = W₂ a₁ + b₂,   ŷ = a₂ = σ(z₂)
```

- $\mathbf{x} \in \mathbb{R}^{n}$ — input vector
- $\mathbf{W}_1 \in \mathbb{R}^{h \times n}$, $\mathbf{b}_1 \in \mathbb{R}^{h}$ — hidden layer weights and bias
- $\mathbf{W}_2 \in \mathbb{R}^{1 \times h}$, $b_2 \in \mathbb{R}$ — output layer weights and bias
- $\sigma(z) = 1 / (1 + e^{-z})$ — sigmoid activation
- $\hat{y} \in (0, 1)$ — predicted probability
- $y \in \{0, 1\}$ — true label

---

## 1. Forward Pass

### Step 1: Hidden layer pre-activation

$$\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$$

In component form: $z_1^{(j)} = \sum_{i=1}^n W_1^{(j,i)} x_i + b_1^{(j)}$ for each hidden neuron $j$.

### Step 2: Hidden layer activation

$$\mathbf{a}_1 = \sigma(\mathbf{z}_1)$$

where $\sigma$ is applied element-wise: $a_1^{(j)} = \sigma(z_1^{(j)})$.

### Step 3: Output layer pre-activation

$$z_2 = \mathbf{W}_2 \mathbf{a}_1 + b_2$$

### Step 4: Output layer activation (prediction)

$$\hat{y} = a_2 = \sigma(z_2)$$

### Step 5: Loss computation

Using Binary Cross-Entropy loss:

$$\mathcal{L}(y, \hat{y}) = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$

---

## 2. Backward Pass

We use the **chain rule** to compute gradients of the loss $\mathcal{L}$ with respect to every parameter.

### 2.1 Gradient of loss w.r.t. output

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}$$

### 2.2 Gradient of loss w.r.t. $z_2$

By the chain rule:

$$\frac{\partial \mathcal{L}}{\partial z_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \sigma'(z_2)$$

We know $\sigma'(z) = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$, therefore:

$$\frac{\partial \mathcal{L}}{\partial z_2} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})} \cdot \hat{y}(1 - \hat{y}) = \hat{y} - y$$

Notice the beautiful simplification: $\boxed{\delta_2 = \hat{y} - y}$. This is the **output error**.

### 2.3 Gradients for output layer parameters

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_2} = \frac{\partial \mathcal{L}}{\partial z_2} \cdot \frac{\partial z_2}{\partial \mathbf{W}_2} = \delta_2 \cdot \mathbf{a}_1^\top$$

$$\frac{\partial \mathcal{L}}{\partial b_2} = \frac{\partial \mathcal{L}}{\partial z_2} = \delta_2$$

### 2.4 Backpropagating to hidden layer

$$\frac{\partial \mathcal{L}}{\partial \mathbf{a}_1} = \frac{\partial \mathcal{L}}{\partial z_2} \cdot \frac{\partial z_2}{\partial \mathbf{a}_1} = \delta_2 \cdot \mathbf{W}_2^\top$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}_1} \cdot \frac{\partial \mathbf{a}_1}{\partial \mathbf{z}_1} = \left( \delta_2 \cdot \mathbf{W}_2^\top \right) \odot \sigma'(\mathbf{z}_1)$$

where $\odot$ is element-wise multiplication. Since $\sigma'(z_1^{(j)}) = a_1^{(j)} (1 - a_1^{(j)})$:

$$\delta_1 = \left( \mathbf{W}_2^\top \delta_2 \right) \odot \mathbf{a}_1 \odot (1 - \mathbf{a}_1)$$

### 2.5 Gradients for hidden layer parameters

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \delta_1 \cdot \mathbf{x}^\top$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \delta_1$$

---

## 3. Parameter Update (Gradient Descent)

$$\mathbf{W}_1 \leftarrow \mathbf{W}_1 - \eta \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{W}_1}$$

$$\mathbf{b}_1 \leftarrow \mathbf{b}_1 - \eta \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{b}_1}$$

$$\mathbf{W}_2 \leftarrow \mathbf{W}_2 - \eta \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{W}_2}$$

$$b_2 \leftarrow b_2 - \eta \cdot \frac{\partial \mathcal{L}}{\partial b_2}$$

where $\eta$ is the learning rate.

---

## Full Vectorized Derivation Summary

| Step | Equation |
|---|---|
| **Forward** | $\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$ |
| | $\mathbf{a}_1 = \sigma(\mathbf{z}_1)$ |
| | $z_2 = \mathbf{W}_2 \mathbf{a}_1 + b_2$ |
| | $\hat{y} = a_2 = \sigma(z_2)$ |
| | $\mathcal{L} = -[y \log \hat{y} + (1 - y) \log(1 - \hat{y})]$ |
| **Backward** | $\delta_2 = \hat{y} - y$ |
| | $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_2} = \delta_2 \cdot \mathbf{a}_1^\top$ |
| | $\frac{\partial \mathcal{L}}{\partial b_2} = \delta_2$ |
| | $\delta_1 = (\mathbf{W}_2^\top \delta_2) \odot \mathbf{a}_1 \odot (1 - \mathbf{a}_1)$ |
| | $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \delta_1 \cdot \mathbf{x}^\top$ |
| | $\frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \delta_1$ |
| **Update** | $\mathbf{W}_1 \leftarrow \mathbf{W}_1 - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}_1}$, etc. |

---

## Why This Matters

- The backward pass computes **exact gradients** for every parameter in $O(n)$ time — same complexity as the forward pass.
- Without backpropagation, we would need $O(n^2)$ forward passes (one per parameter) to approximate gradients numerically.
- **Vanishing gradients**: Notice $\delta_1$ contains the factor $\mathbf{a}_1 \odot (1 - \mathbf{a}_1)$. Since each $a_1^{(j)} \in (0, 1)$, this product is at most $0.25$, shrinking gradients as depth increases.

---

**Next:** [04 - Loss Functions.md](./04%20-%20Loss%20Functions.md)
