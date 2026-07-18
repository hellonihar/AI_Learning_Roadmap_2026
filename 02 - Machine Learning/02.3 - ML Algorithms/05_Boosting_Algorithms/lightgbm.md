# LightGBM

**Light Gradient Boosting Machine** — a gradient boosting framework designed for **speed and efficiency** on large datasets.

## What Makes LightGBM Different

### Leaf-Wise Tree Growth
Unlike level-wise growth (XGBoost), LightGBM grows the tree **leaf-wise** — it only splits the leaf with the highest loss reduction.

```
Level-wise (XGBoost):        Leaf-wise (LightGBM):
    ┌───┐                        ┌───┐
   ┌┘   └┐                      ┌┘   └┐
  ┌┘     └┐                     │     └┐
 ┌┘       └┐                    │      └┐
                                  │       └┐
                                  │        └┐
```

Leaf-wise grows deeper trees faster but can overfit on small data.

### GOSS (Gradient-based One-Side Sampling)
Instead of using all data points, LightGBM keeps all high-gradient points (large errors) and randomly samples low-gradient points. This focuses computation on the hardest cases.

### EFB (Exclusive Feature Bundling)
Groups mutually exclusive features (e.g., one-hot encoded columns that are never simultaneously 1) into a single feature. Reduces dimensionality.

## Key Hyperparameters

| Parameter | Effect |
|---|---|
| `num_leaves` | Max leaves per tree (default 31) — controls complexity |
| `learning_rate` (eta) | Shrinkage rate |
| `min_data_in_leaf` | Minimum samples per leaf — prevents overfitting |
| `feature_fraction` | Feature sampling per iteration |
| `bagging_fraction` | Row sampling |
| `max_bin` | Max discretization bins — lower = faster but less accurate |

## When to Use

- Very large datasets (>100K samples)
- When training speed is critical
- High-dimensional sparse features (one-hot encoded)
- Memory-constrained environments

## Examples

1. **Click-through rate (CTR) prediction**: Billions of ad impressions with millions of sparse features. LightGBM trains 10x faster than XGBoost on this data, enabling daily retraining.
2. **Real-time bidding**: Ad auction systems need sub-second model prediction. LightGBM's small model size and fast inference make it suitable for latency-critical systems.
3. **IoT sensor data**: Millions of sensor readings per day, thousands of devices. LightGBM handles the scale efficiently, retraining nightly on the full dataset.
