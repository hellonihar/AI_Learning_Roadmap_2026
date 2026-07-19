# Learning Rate Scheduling

The learning rate $\eta$ is often the most important hyperparameter. Rather than using a fixed value, scheduling adjusts $\eta$ during training to balance early exploration with late convergence.

## Step Decay

Reduce $\eta$ by a factor $\gamma$ every $k$ epochs:

$$ \eta_t = \eta_0 \cdot \gamma^{\lfloor t / k \rfloor} $$

Common: $\gamma = 0.1$, $k = 30$ epochs.

**When to use**: Legacy computer vision (ResNet schedules). Simple and effective when you know the optimal number of steps.

## Exponential Decay

Continuous decay at each step:

$$ \eta_t = \eta_0 \cdot e^{-kt} $$

**When to use**: When you want smooth, monotonic decay. Often used with SGD.

## Cosine Annealing

Smooth decrease following a cosine curve (Loshchilov & Hutter, 2016):

$$ \eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right) $$

**When to use**: Default for modern training. Works well with SGD+Momentum. Can be combined with warmup. Warm restarts (SGDR) periodically reset $\eta$ for escaping local minima.

## Warmup

Linear increase from 0 to $\eta_0$ over $W$ steps:

$$ \eta_t = \eta_0 \cdot \min\left(1, \frac{t}{W}\right) $$

**When to use**: Essential for Adam/AdamW with Transformers. Prevents early large updates when gradient statistics are poorly estimated. Typically $W = 4000$–$10000$ steps for large language models.

## ReduceLROnPlateau

Reduce $\eta$ by $\gamma$ when a metric (e.g., validation loss) plateaus for $p$ epochs:

```python
if val_loss_not_improved_for(p_epochs):
    eta *= gamma
```

**When to use**: When you don't know the total training length. Automatically adapts to convergence.

## Cyclic LR

Cosine annealing with periodic reset from $\eta_{\min}$ to $\eta_{\max}$ (Smith, 2017):

$$ \eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t \bmod T}{T}\pi\right)\right) $$

**When to use**: Rapid exploration of the loss landscape. Can find good solutions faster than monotonic schedules.

## Summary

| Schedule | Best for |
|---|---|
| Step Decay | Known training length, CNN baselines |
| Exponential Decay | Smooth decay needs |
| Cosine Annealing | SGD+Momentum, modern image models |
| Warmup | Adam/AdamW, Transformers |
| ReduceLROnPlateau | Unknown optimal length, auto-tuning |
| Cyclic LR | Fast experimentation, hyperparameter search |
