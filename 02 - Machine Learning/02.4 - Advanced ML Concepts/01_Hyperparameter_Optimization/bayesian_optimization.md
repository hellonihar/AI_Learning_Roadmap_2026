# Bayesian Optimization

An **intelligent** hyperparameter search that learns from previous trials to focus on promising regions.

## How It Works

1. Build a **surrogate model** (usually Gaussian Process or Tree-structured Parzen Estimator) that approximates the objective function
2. Use an **acquisition function** to decide which hyperparameter combination to try next — balancing exploration (try uncertain regions) and exploitation (try regions known to be good)
3. Evaluate the chosen combination
4. Update the surrogate model with the new result
5. Repeat

```
Trial 1: random point → score 0.82
Trial 2: random point → score 0.76
Trial 3: near Trial 1 (exploit) → score 0.85
Trial 4: uncertain region (explore) → score 0.71
Trial 5: near Trial 3 (exploit) → score 0.87
...
```

## Grid vs Random vs Bayesian

| | Grid | Random | Bayesian |
|---|---|---|---|
| **Intelligent?** | No | No | Yes (learns from history) |
| **Efficiency** | Lowest | Moderate | Highest |
| **Trials to find optimum** | Many | Moderate | Few (typically 3-5× fewer than random) |
| **Parallelizable?** | Trivially | Trivially | Limited (sequential updates) |
| **Implementation complexity** | None | None | Moderate (needs library) |

## Key Components

### Surrogate Model
- **Gaussian Process (GP)**: Works well for continuous hyperparameters (<20 dimensions)
- **TPE (Tree Parzen Estimator)**: Better for high-dimensional or mixed categorical/continuous spaces

### Acquisition Function
- **Expected Improvement (EI)**: How much better than current best do we expect this point to be?
- **Upper Confidence Bound (UCB)**: Trade-off between predicted performance and uncertainty
- **Probability of Improvement (PI)**: Probability that this point beats the current best

## When to Use

- Expensive training (deep learning, large datasets) — minimize wasted trials
- 5-20 hyperparameters
- Continuous hyperparameters (learning rate, regularization)
- When compute budget is severely limited

## Examples

1. **Deep learning hyperparameter tuning**: Training a ResNet takes hours. Grid search with 100 combinations is infeasible. Bayesian optimization finds a good configuration in 20-30 trials — 5× fewer than random search. Libraries like Optuna, Hyperopt, or Ray Tune handle this automatically.
2. **Gradient boosting on large data**: XGBoost on 1M rows takes 5 minutes per training. Grid (100 combos) = 8 hours. Bayesian (20 trials guided by past results) = 1.5 hours. The Bayesian result is typically within 1-2% of the grid optimum.
3. **NLP model tuning**: Fine-tuning BERT with different learning rates, batch sizes, and warmup steps. Each trial costs GPU time. Bayesian optimization with early stopping (prune bad trials early using learning curves) reduces total search time by 80%.
