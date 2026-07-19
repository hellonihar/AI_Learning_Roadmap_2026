# Hyperparameter Tuning

Hyperparameters (LR, batch size, weight decay, dropout rate, architecture choices) cannot be learned from data — they must be set before training. Effective tuning is essential for state-of-the-art results.

## Grid Search

Exhaustively evaluates all combinations of a predefined set of values.

```python
lrs = [1e-3, 1e-4, 1e-5]
weight_decays = [1e-4, 1e-5]
for lr in lrs:
    for wd in weight_decays:
        train(lr=lr, weight_decay=wd)
```

**Pros**: Simple, parallelizable, reproducible.
**Cons**: Curse of dimensionality — $k$ hyperparameters with $n$ values each requires $n^k$ runs. Inefficient — wastes trials on unimportant hyperparameters.

## Random Search

Samples hyperparameters uniformly from specified distributions (Bergstra & Bengio, 2012).

```python
for _ in range(N):
    lr = 10**uniform(-5, -1)
    wd = 10**uniform(-5, -2)
    train(lr=lr, weight_decay=wd)
```

**Pros**: More efficient than grid search in high dimensions. Each trial explores a unique combination. Can use log-uniform for scale parameters.
**Cons**: No memory of past trials. Wastes trials in poor regions.

**Key insight**: Random search is strictly better than grid search because not all hyperparameters matter equally. Random search explores more distinct values per important hyperparameter for the same budget.

## Bayesian Optimization

Builds a probabilistic surrogate model (typically a Gaussian Process or Tree-Structured Parzen Estimator) mapping hyperparameters to validation performance. An acquisition function (Expected Improvement, UCB) selects the next promising candidate.

### Gaussian Processes (GP)

Surrogate: $f(h) \sim \mathcal{GP}(\mu(h), k(h, h'))$ where $h$ are hyperparameters. The GP provides a mean prediction and uncertainty for any point.

Acquisition: Upper Confidence Bound $h_{t+1} = \arg\max_h \mu_t(h) + \kappa \sigma_t(h)$

### Tree-Structured Parzen Estimator (TPE)

Non-parametric alternative to GPs. Models $p(y|x)$ and $p(x|y)$ using kernel density estimates. Used in Hyperopt and Optuna.

**Pros**: Sample-efficient — requires 10–30x fewer trials than random search. Handles conditional hyperparameter spaces.
**Cons**: Sequential (hard to parallelize). Overheads for high-dimensional spaces. Requires careful initialization.

## Population-Based Training (PBT)

Inspired by genetic algorithms. Trains a population of models in parallel, periodically copying weights from high-performing "workers" to low-performing ones (or mutating hyperparameters).

```python
# Each worker trains with (lr, wd, dropout)
# Every K steps:
#   Eval all workers on validation set
#   Bottom 25%: copy weights from top 25%, mutate hyperparams
#   Continue training
```

**Pros**: Hyperparameters evolve during training (LR schedules, etc.). Highly parallel.
**Cons**: Complex infrastructure, requires many GPUs.

## Practical Workflow

1. **Start with manual tuning** — find a reasonable LR range and confirm the model can overfit one batch.
2. **Random search** — 20–50 trials on LR, weight decay, dropout. Use log-uniform for scale params.
3. **Refine with Bayesian Optimization** — 50–100 more trials around the best region.
4. **During production**: Sweep LR schedule parameters + optimizer params.
5. **For large-scale**: Use PBT with a population of 32–64.

### Tools

- **Optuna** (Python) — TPE-based, pruning, easy API
- **Ray Tune** — Distributed, supports PBT, ASHA
- **Weights & Biases Sweeps** — Cloud-based, Bayesian + grid + random
- **Hyperopt** — TPE, supports conditional spaces
