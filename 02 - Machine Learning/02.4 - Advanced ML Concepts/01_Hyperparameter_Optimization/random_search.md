# Random Search

Samples hyperparameter combinations **randomly** from defined distributions instead of trying all combinations.

## How It Works

Define a distribution for each hyperparameter (uniform, log-uniform, etc.). Sample N random combinations. Train and evaluate each. Pick the best.

```
n_iter = 50
max_depth ~ Uniform(3, 15)
n_estimators ~ Uniform(50, 500)
learning_rate ~ LogUniform(0.001, 0.3)
```

## Why Random Search Often Beats Grid

Random search is more **efficient** because not all hyperparameters matter equally. Grid search wastes trials on irrelevant dimensions while random search explores more values of important ones.

| Dimension | Grid Search | Random Search (60 trials) |
|---|---|---|
| Important HP (learning_rate) | 3 values (0.01, 0.1, 0.3) | 60 unique values |
| Unimportant HP (max_depth) | 3 values | 60 unique values |
| **Total useless combos** | Explores every bad depth | Wastes fewer trials on irrelevant params |

## Key Hyperparameter: n_iter

- Small (10-30): Quick exploration, may miss optimum
- Medium (50-100): Good balance for most problems
- Large (100-500): Thorough, diminishing returns

## When to Use

- More than 3-4 hyperparameters
- When compute budget is limited
- As a first pass before Bayesian optimization
- When you don't know which hyperparameters are important

## Examples

1. **XGBoost tuning (7+ hyperparameters)**: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`. Grid would need thousands of combinations. Random search with 100 trials covers the space efficiently and usually finds near-optimal settings.
2. **Neural network tuning**: `learning_rate`, `batch_size`, `num_layers`, `layer_size`, `dropout`, `optimizer`, `activation`. These interact in complex ways — grid search is impractical. Random search with 50-100 trials identifies good regions.
3. **CatBoost tuning**: 10+ hyperparameters including `depth`, `learning_rate`, `l2_leaf_reg`, `border_count`, `bagging_temperature`. Random search with early stopping finds a good configuration in <50 trials.
