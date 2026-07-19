# Loss Functions & Optimizers

Choosing the right loss function and optimizer is critical for training success. PyTorch provides a comprehensive suite of both.

## Loss Functions (`torch.nn`)

### Regression

```python
nn.MSELoss()                          # Mean Squared Error: (y - y_hat)²
nn.L1Loss()                           # Mean Absolute Error: |y - y_hat|
nn.SmoothL1Loss()                     # Huber-like, less sensitive to outliers
```

- **MSELoss** — standard for regression. Sensitive to outliers.
- **L1Loss** — robust to outliers, slower convergence.
- **SmoothL1Loss** — behaves like L1 far from zero, L2 near zero. Used in object detection (Smooth L1).

### Classification

```python
nn.CrossEntropyLoss()                 # LogSoftmax + NLLLoss combined
nn.BCELoss()                          # Binary cross-entropy (requires sigmoid)
nn.BCEWithLogitsLoss()                # Sigmoid + BCE in one step (preferred)
nn.NLLLoss()                          # Negative log-likelihood (after LogSoftmax)
```

| Task | Output Activation | Loss |
|------|-------------------|------|
| Multi-class | None (raw logits) | `CrossEntropyLoss` |
| Multi-label / Binary | None (raw logits) | `BCEWithLogitsLoss` |
| Multi-class (custom) | `LogSoftmax` | `NLLLoss` |

**Why `BCEWithLogitsLoss` over `BCELoss`?** Combining sigmoid + BCE numerically stabilizes the gradient. Using raw `BCELoss` requires manually adding a sigmoid layer, which can produce `log(0)` — `BCEWithLogitsLoss` avoids this.

### Other Common Losses

```python
nn.KLDivLoss()                        # KL divergence (for distillation)
nn.MarginRankingLoss()                # Ranking tasks
nn.TripletMarginLoss()                # Metric learning
nn.CosineEmbeddingLoss()              # Similarity learning
nn.PoissonNLLLoss()                   # Count data
```

## Optimizers (`torch.optim`)

```python
import torch.optim as optim
```

### SGD (Stochastic Gradient Descent)

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
```

The classic. **Momentum** accelerates convergence through valleys. **Nesterov** momentum (`nesterov=True`) looks ahead. Good for CV, works well with proper LR tuning.

### Adam

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
```

Adaptive learning rate per parameter. Combines momentum (first moment) and RMSProp (second moment). Default go-to for most deep learning tasks. Less sensitive to LR tuning than SGD.

### AdamW

```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

Adam with **decoupled weight decay** — weight decay is applied directly to parameters rather than through the gradient. Theoretically correct regularization. Used by most modern transformer training recipes (LLMs, ViTs).

### RMSProp

```python
optimizer = optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99)
```

Maintains per-parameter learning rates based on the moving average of squared gradients. Works well for RNNs. Less common now but still useful.

### Optimizer Comparison

| Optimizer | Adaptive LR | Weight Decay | Best For |
|-----------|-------------|--------------|----------|
| SGD | No | Integrated | CV, well-tuned schedules |
| Adam | Yes | Integrated | General purpose, text, audio |
| AdamW | Yes | Decoupled | Transformers, LLMs, modern architectures |
| RMSProp | Yes | Integrated | RNNs, online learning |

## Learning Rate Schedulers

```python
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
```

### StepLR

```python
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# LR drops by 0.1× every 10 epochs
```

Simple and effective for many tasks.

### ReduceLROnPlateau

```python
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
# Reduces LR by half when validation loss plateaus for 5 epochs
```

Adaptive — no fixed schedule. Must call `.step(val_loss)` with the monitored metric.

### CosineAnnealingLR

```python
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
# LR follows a cosine curve from initial_lr to eta_min over T_max epochs
```

Popular in modern training (used in ResNets, transformers). Often combined with warmup.

### CosineAnnealingWarmRestarts

```python
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
# Cosine schedule with periodic restarts, doubling period each cycle
```

### Putting It Together

```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(50):
    train_one_epoch()
    scheduler.step()               # update LR
```

Or with plateau-based scheduling:

```python
scheduler = ReduceLROnPlateau(optimizer, patience=3)

for epoch in range(100):
    train_loss = train_one_epoch()
    val_loss = validate()
    scheduler.step(val_loss)       # monitors validation loss
```

## Tips

- **Start with Adam** (lr=3e-4) for quick prototyping, then switch to AdamW for production.
- **Weight decay** is essential for generalization — use 1e-4 to 1e-2.
- **LR scheduling** matters as much as optimizer choice — a good scheduler can recover a bad LR.
- **Warmup** (linear increase for first few epochs) helps stabilize large-batch and transformer training.

> **Summary:** Match your loss to your task (CrossEntropy for classification, MSE for regression). Adam/AdamW covers 90% of use cases. Add scheduling as soon as you have a working loop.
