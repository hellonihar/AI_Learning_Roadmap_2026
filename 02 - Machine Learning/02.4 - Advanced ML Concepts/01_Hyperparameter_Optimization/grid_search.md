# Grid Search

Systematically tries every combination of hyperparameters from a predefined grid.

## How It Works

Define a grid of hyperparameter values. Train and evaluate a model for every combination. Pick the combination with the best validation score.

```
grid = {
    'max_depth': [3, 5, 7, 10],
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3]
}
# Total combinations: 4 × 3 × 3 = 36
```

## Pros and Cons

| Pros | Cons |
|---|---|
| Guaranteed to find the best combination in the grid | Curse of dimensionality — combinations explode exponentially |
| Simple, interpretable, reproducible | Wasteful — spends equal time on good and bad regions |
| Easy to parallelize (each combination is independent) | Doesn't learn from previous trials |

## When to Use

- Small number of hyperparameters (< 4)
- Small hyperparameter ranges
- Sufficient compute budget
- As a baseline before trying more advanced methods

## Examples

1. **Random Forest tuning**: Grid over `n_estimators=[100, 200, 500]`, `max_depth=[5, 10, None]`, `min_samples_leaf=[1, 2, 5]`. 3×3×3 = 27 combinations. Each trains in 30 seconds. Total: ~13 minutes. Straightforward and complete.
2. **SVM tuning**: Grid over `C=[0.1, 1, 10, 100]`, `gamma=[0.001, 0.01, 0.1, 1]`, `kernel=['rbf']`. 4×4 = 16 combinations. SVM training is O(n²) so grid search may be expensive, but for small datasets it's fine.
3. **XGBoost initial tuning**: Coarse grid first: `max_depth=[3, 6, 9]`, `learning_rate=[0.01, 0.1, 0.3]`, `subsample=[0.6, 0.8, 1.0]`. 3×3×3 = 27 combos. Identify promising region, then fine-tune with random search or Bayesian optimization.
