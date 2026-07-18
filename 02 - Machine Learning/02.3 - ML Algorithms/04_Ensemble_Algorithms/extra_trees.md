# Extra Trees (Extremely Randomized Trees)

Similar to Random Forest but with **more randomness** — random split thresholds instead of optimal ones.

## How It Works

| | Random Forest | Extra Trees |
|---|---|---|
| Bootstrap sample | Yes | No (trains on full data) |
| Split threshold | Optimal (best split) | **Random threshold** |
| Feature selection | Random subset per split | Random subset per split |
| Variance reduction | Moderate | **Higher** |
| Bias | Moderate | Slightly higher |

By randomizing the split thresholds, individual trees become even more different from each other → lower correlation → averaging reduces variance more effectively.

## When to Use Extra Trees Over Random Forest

- Very high-dimensional data: random splits are computationally cheaper than finding optimal splits
- When you need even lower variance than RF (at the cost of slightly higher bias)
- As a computationally cheaper alternative to RF (no need to evaluate all possible split points)

## Examples

1. **High-dimensional bioinformatics**: 10K gene features, 500 samples. Extra Trees outperforms Random Forest because the random split thresholds add diversity that helps in very high dimensions.
2. **Real-time sensor classification**: 500 sensors, streaming data. Extra Trees trains faster (random splits vs optimal splits), critical when the model needs periodic retraining on new data.
3. **Feature selection preprocessing**: Extra Trees feature importance is often more reliable than RF for ranking thousands of features before passing to a secondary model.
