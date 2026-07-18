# One-Class SVM

A **novelty detection** algorithm that learns a boundary around normal data points.

## How It Works

One-Class SVM finds a **tight boundary** that encloses most training data points. Points outside the boundary are anomalies.

- Training: uses only **normal data** (no anomalies in the training set)
- Predicts: whether a new point is within the normal boundary

This is **novelty detection** — you assume the training data is clean and want to detect new outliers. This differs from **outlier detection** (like Isolation Forest) where training data already contains anomalies.

## Key Hyperparameters

| Parameter | Effect |
|---|---|
| **ν (nu)** | Upper bound on the fraction of training errors and lower bound on support vectors. 0.1 = expect ~10% anomalies. Smooths the boundary. |
| **γ (gamma)** | RBF kernel coefficient. High γ = tight boundary around each point (may overfit). Low γ = smooth boundary (may underfit). |

## When to Use One-Class SVM

- You have **clean normal data** for training (no anomalies in training set)
- Small to medium datasets (<10K points)
- The boundary around normal data is expected to be non-linear
- Need probability-like scores (distance from boundary)

## Limitations

- **Slow** on large datasets (O(n²) complexity)
- Sensitive to **ν and γ** — needs hyperparameter tuning
- Doesn't scale well to **high dimensions** (>100 features)
- Training data must be **representative** of normal (not contaminated)

## Examples

1. **New product quality control**: Training on 5K known-good units from the production line (verified by human inspectors). One-Class SVM learns the boundary of "normal operation." New units far from this boundary are flagged for inspection.
2. **Server monitoring (novelty detection)**: Train only on normal server behavior data. Any new behavior that falls outside the learned boundary flags a potential intrusion or system failure before it causes downtime.
3. **Medical screening**: Train on healthy patient data (confirmed healthy). New patient whose vitals fall outside the "healthy boundary" is flagged for further testing. This is a screening tool, not a diagnostic — it catches unusual cases but doesn't classify them.
