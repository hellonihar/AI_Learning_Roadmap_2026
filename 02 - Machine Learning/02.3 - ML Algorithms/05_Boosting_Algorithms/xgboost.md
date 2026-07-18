# XGBoost

**Extreme Gradient Boosting** — the most popular gradient boosting implementation. Optimized for speed and performance.

## What Makes XGBoost Special

| Feature | Why It Matters |
|---|---|
| **Regularization** | L1 + L2 on leaf weights — prevents overfitting beyond standard GB |
| **Tree pruning** | Grows trees depth-first, prunes backward — more efficient than stopping criteria |
| **Weighted quantile sketch** | Efficient handling of weighted data |
| **Sparsity awareness** | Learns optimal direction for missing values |
| **Column block** | Pre-sorts features for parallel split finding |
| **Cache-aware access** | Optimized memory access patterns |

## Key Hyperparameters

| Parameter | Effect |
|---|---|
| `n_estimators` | Number of rounds (increase with low LR) |
| `learning_rate` (eta) | Step size shrinkage (default 0.3) |
| `max_depth` | Tree depth (default 6) — deeper = more complex |
| `subsample` | Row sampling ratio (0.5-1.0) |
| `colsample_bytree` | Feature sampling per tree |
| `gamma` (min_split_loss) | Minimum loss reduction to split further |
| `reg_alpha` (L1) | L1 regularization on leaf weights |
| `reg_lambda` (L2) | L2 regularization on leaf weights |

## When to Use

- Medium-to-large tabular datasets
- When you need state-of-the-art performance on structured data
- When training speed matters (faster than sklearn's GradientBoosting)

## Examples

1. **Kaggle competition winning**: The majority of Kaggle tabular competition winners use XGBoost (or LightGBM). It consistently outperforms other algorithms on diverse datasets — from house prices to customer churn to insurance claims.
2. **Ad click prediction**: XGBoost handles billions of training samples (distributed mode) with hundreds of features. Its sparsity-aware algorithm efficiently handles the sparse one-hot encoded features common in CTR prediction.
3. **Credit risk modeling**: XGBoost with regularization produces stable, high-performance models that pass regulatory scrutiny. Feature importance and SHAP values provide interpretability.
