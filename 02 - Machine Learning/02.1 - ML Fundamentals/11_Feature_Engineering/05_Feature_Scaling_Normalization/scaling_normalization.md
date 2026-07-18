# Feature Scaling & Normalization

## Why Scale Features?

Features with different units and magnitudes cause problems for many models.

| Feature | Range | Issue |
|---|---|---|
| Income | 0 - \$10M | Dominates distance calculations |
| Age | 0 - 100 | Gets ignored compared to income |
| Credit score | 300 - 850 | Middle ground |

- **Gradient descent**: Converges faster when all features are on a similar scale
- **Distance-based models (k-NN, SVM, k-means)**: Features with large values dominate the distance calculation
- **Regularization (L1/L2)**: Penalizes all coefficients equally — a \$1 change in income vs a \$1 change in age have entirely different meanings

**When scaling is unnecessary**: Tree-based models (decision trees, random forest, gradient boosting) — they split on feature thresholds independently of scale. Scaling doesn't change tree splits.

## Feature Magnitude vs Importance

Larger magnitude ≠ more important. A feature with range [0, 1B] will dominate simply because its numbers are bigger, not because it's more predictive. Scaling removes this artifact.

## Standardization (Z-Score)

`x_scaled = (x — mean) / std`

- Centers at 0, standard deviation = 1
- No fixed range — values can exceed [-3, 3] for outliers
- **Best for**: Data with Gaussian distribution, models that assume normally distributed features

**Example**: Income distribution (mean=$60K, std=$40K). Income of \$200K → z-score = (200,000 - 60,000) / 40,000 = 3.5. The value 3.5 tells the model "this is 3.5 standard deviations above average."

## Min-Max Scaling

`x_scaled = (x — min) / (max — min)`

- Scales to [0, 1] (or [-1, 1] if you adjust)
- **Sensitive to outliers** — one extreme value compresses all other values into a tiny range
- **Best for**: Neural networks (often expect [0,1] inputs), known bounded ranges

**Example**: House prices (\$100K — \$10M). Min-max scaling: a \$500K house → (500K - 100K) / (10M - 100K) = 0.04. If one mansion costs \$100M, all other houses get squished near 0.

## Robust Scaling

`x_scaled = (x — median) / IQR`

- Uses median and interquartile range (25th-75th percentile)
- **Not affected by outliers** — median and IQR are robust to extreme values
- **Best for**: Data with many outliers

**Example**: Transaction amounts — most are 10-100, a few are 10,000+. Robust scaling keeps the common range well-separated while outliers are still detectable.

## When Scaling Is Unnecessary

| Model | Needs Scaling | Reason |
|---|---|---|
| Linear / Logistic Regression | ✅ | Gradient descent, coefficient interpretation |
| SVM | ✅ | Distance-based margin |
| k-NN, k-Means | ✅ | Distance-based algorithms |
| Neural Networks | ✅ | Gradient descent, activation functions |
| PCA | ✅ | Variance-based — large magnitude dominates |
| Decision Trees / RF / XGBoost | ❌ | Tree splits on thresholds, unaffected by scale |
| Naive Bayes | ❌ | Handles distributions, not distances |

## Scaling Before vs After Train-Test Split (Very Important)

**Rule**: Fit the scaler **on the training set only**. Apply the **same** transformation to val/test sets.

**Correct:**
```
scaler.fit(X_train)            # compute mean/std from training only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)    # use training statistics
```

**Wrong:**
```
scaler.fit(X_full)             # test data influences training statistics
X_full_scaled = scaler.transform(X_full)
# Then split — contaminated!
```

**Example**: If test set contains an extreme outlier (e.g., a \$100M house), min-max scaling on full data shifts all training values because the max increased. The model implicitly "knows" about the test set's extreme values. Always fit on train, transform test.

## Summary

```
Standardization:  (x - mean) / std        → unbounded, handles outliers OK
Min-Max Scaling:  (x - min) / (max - min) → bounded [0,1], sensitive to outliers
Robust Scaling:   (x - median) / IQR      → unbounded, outlier-resistant
```
