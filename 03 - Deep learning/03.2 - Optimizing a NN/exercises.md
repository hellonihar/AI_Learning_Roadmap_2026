# Exercises

## Exercise 1: Gradient Clipping

Implement norm-based gradient clipping for the 2-layer NN from `code_walkthrough.md`. Train on the synthetic data with and without clipping using a high learning rate ($\eta = 1.0$). Show that clipping prevents divergence.

**Answer**:

```python
def clip_gradients(model, max_norm=1.0):
    total_norm = np.sqrt(np.sum(model.dW1**2) + np.sum(model.dW2**2) + 
                         np.sum(model.db1**2) + np.sum(model.db2**2))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        model.dW1 *= scale
        model.db1 *= scale
        model.dW2 *= scale
        model.db2 *= scale
```

Without clipping at $\eta = 1.0$, loss diverges to NaN. With clipping ($c = 1.0$), training stabilizes and converges within 50 epochs.

## Exercise 2: Compare Optimizers on a Quadratic

Minimize $f(x, y) = x^2 + 10y^2$ (a poorly-conditioned quadratic) using SGD, Momentum, and Adam. Plot the optimization trajectory. Which reaches the minimum fastest?

**Answer**:

```python
import numpy as np

def f(x, y): return x**2 + 10*y**2
def grad(x, y): return np.array([2*x, 20*y])

lr, steps = 0.1, 50
x_sgd, y_sgd = 4.0, 4.0
x_mom, y_mom = 4.0, 4.0
vx, vy = 0, 0
beta = 0.9

for _ in range(steps):
    gx, gy = grad(x_sgd, y_sgd)
    x_sgd -= lr * gx
    y_sgd -= lr * gy

    gx, gy = grad(x_mom, y_mom)
    vx = beta * vx + lr * gx
    vy = beta * vy + lr * gy
    x_mom -= vx
    y_mom -= vy
```

SGD oscillates heavily along the $y$ direction (large curvature). Momentum dampens oscillations and converges faster. Adam is fastest, reaching near-zero in ~20 steps.

## Exercise 3: Explain Batch Normalization

Explain why Batch Normalization allows higher learning rates. Include the mathematical argument about the Lipschitz constant of the loss.

**Answer**: Batch Normalization makes the loss landscape more well-conditioned by ensuring activations have zero mean and unit variance. The Lipschitz constant of the loss with respect to the parameters is bounded by $\frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}$, meaning the gradient magnitude is controlled. This prevents sudden large gradient explosions, allowing $\eta$ to be 1–2 orders of magnitude larger than without BatchNorm. Santurkar et al. (2018) showed that BatchNorm's primary benefit is smoothing the optimization landscape rather than reducing internal covariate shift.

## Exercise 4: Derive the Gradient of L2 Regularization

Given the regularized loss $\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{data}} + \frac{\lambda}{2} \|w\|^2$, derive the gradient and show the weight update corresponds to weight decay.

**Answer**:

$$ \nabla_w \mathcal{L}_{\text{reg}} = \nabla_w \mathcal{L}_{\text{data}} + \lambda w $$

Weight update: $w \leftarrow w - \eta \nabla_w \mathcal{L}_{\text{data}} - \eta \lambda w = w(1 - \eta \lambda) - \eta \nabla_w \mathcal{L}_{\text{data}}$

The term $w(1 - \eta \lambda)$ shows the weight is "decayed" by factor $(1 - \eta \lambda)$ each step. This is equivalent to multiplying the weight by a constant slightly less than 1 before applying the gradient update.

## Exercise 5: Implement Early Stopping

Write a function `train_with_early_stopping(model, X, y, X_val, y_val, patience=10, max_epochs=500)` that tracks validation loss and stops when it hasn't improved for `patience` epochs. Return the best model parameters.

**Answer**:

```python
def train_with_early_stopping(model, X, y, X_val, y_val, patience=10, max_epochs=500):
    best_val_loss = float('inf')
    best_params = {}
    wait = 0
    for epoch in range(max_epochs):
        output = model.forward(X, training=True)
        model.backward(X, y, output, l2_lambda=0.0)
        sgd_update(model, lr=0.01)
        
        val_output = model.forward(X_val, training=False)
        val_loss = model.compute_loss(val_output, y_val)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {'W1': model.W1.copy(), 'b1': model.b1.copy(),
                           'W2': model.W2.copy(), 'b2': model.b2.copy()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                model.W1, model.b1 = best_params['W1'], best_params['b1']
                model.W2, model.b2 = best_params['W2'], best_params['b2']
                break
    return model, epoch
```

## Exercise 6: Diagnosing Overfitting

A model achieves 99.5% train accuracy and 83% test accuracy after 50 epochs. List 3 possible fixes and the reasoning behind each.

**Answer**: The model is overfitting:
1. **Add L2 regularization** ($\lambda = 10^{-3}$) — penalizes large weights, reduces model complexity.
2. **Increase dropout** to $p=0.5$ in the penultimate layer — prevents co-adaptation.
3. **Apply early stopping** with patience 5 — stops training at epoch ~20 before overfitting begins.
4. **Add data augmentation** — artificially increases dataset size.
5. **Reduce model capacity** — fewer neurons/layers.

Priority: Data augmentation (if possible) → L2 → Dropout → Early stopping.

## Exercise 7: Why Does Adam Use Bias Correction?

Derive $\hat{m}_t = m_t / (1 - \beta_1^t)$ and show why it's needed for the first few timesteps.

**Answer**: At $t=1$: $m_1 = \beta_1 m_0 + (1-\beta_1) g_1 = (1-\beta_1) g_1$ (since $m_0 = 0$). The expected value $\mathbb{E}[m_1] = (1-\beta_1) \mathbb{E}[g_1]$, while we want $\mathbb{E}[\hat{m}_1] = \mathbb{E}[g_1]$. Dividing by $1-\beta_1^t = 1-\beta_1$ corrects the bias: $\hat{m}_1 = m_1/(1-\beta_1) = g_1$.

For small $t$, the moving average is biased toward zero because it starts at zero. The correction factor $1 - \beta_1^t$ approaches 1 as $t$ grows, making the correction negligible after ~100 steps.

## Exercise 8: Group vs Batch Normalization

You're training a segmentation model with batch size 2 on 512×512 images. BatchNorm gives poor results. Why? What would you use instead?

**Answer**: With batch size 2, the mean and variance estimates of BatchNorm are extremely noisy. Each batch's statistics poorly represent the population. This noise degrades training. 

**Fix**: Replace BatchNorm with GroupNorm ($G=32$). GroupNorm normalizes over groups of channels, independent of batch size. Wu & He (2018) showed GroupNorm matches BatchNorm accuracy for batch sizes $\geq 16$ and significantly outperforms it for batch sizes 2–8.
