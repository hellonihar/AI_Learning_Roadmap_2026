# k-Nearest Neighbors (kNN)

Classifies a point based on the **majority vote of its k nearest neighbors**. No training — just memorizes the data.

## How It Works

1. Store all training data
2. For a new point, find the k closest training points (by distance)
3. Classify by majority vote (classification) or average (regression)

## Choosing k

| k | Effect |
|---|---|
| k = 1 | Perfect on training, terrible on test (high variance) |
| k = small (3-5) | Flexible, moderate overfitting risk |
| k = moderate (10-50) | Smooth decision boundary, good generalization |
| k = N (all data) | Always predicts the majority class (high bias) |

**Rule**: Start with k = √N. Use cross-validation to tune.

## Distance Metrics

| Metric | Best For |
|---|---|
| Euclidean | Continuous features, same scale |
| Manhattan | High-dimensional data |
| Cosine | Text data, sparse vectors |

## Curse of Dimensionality

As dimensions increase, all points become equally far apart. kNN breaks down beyond ~20 features. Use dimensionality reduction or feature selection first.

## Examples

1. **Handwriting recognition**: 8×8 pixel images (64 features). kNN with k=3 achieves ~97% accuracy on digits. Simple, competitive with neural networks for small datasets.
2. **Recommendation (user-based)**: Find k users with most similar rating patterns, recommend items they liked that the target user hasn't seen.
3. **Anomaly detection**: Points whose k nearest neighbors are far away are anomalies. Used in credit card fraud as a baseline.
