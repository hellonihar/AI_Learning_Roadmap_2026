# Random Forest

An ensemble of decision trees trained on **bootstrap samples** with **random feature subsets**. The wisdom of the crowd: many imperfect trees collectively make excellent predictions.

## How It Works

1. **Bootstrap sampling**: Create N random subsets of the training data (sampled with replacement)
2. **Random feature selection**: At each split, consider only a random subset of features (typically √p for classification, p/3 for regression)
3. **Train one tree per bootstrap sample**
4. **Average predictions** (regression) or **majority vote** (classification)

## Why It Works

Individual trees overfit and have high variance. But their errors are uncorrelated (because of random sampling and random features). Averaging uncorrelated errors dramatically reduces variance without increasing bias.

## Key Hyperparameters

| Parameter | Effect | Typical Range |
|---|---|---|
| n_estimators | More trees = more stable, diminishing returns | 100-1000 |
| max_depth | Tree complexity | None (full trees) or 10-50 |
| min_samples_leaf | Prevent overfitting | 1-5 |
| max_features | Feature randomness per split | sqrt(p) or log2(p) |

## Out-of-Bag (OOB) Score

Each tree trains on ~63% of data (bootstrap sampling). The remaining ~37% (out-of-bag) serves as a built-in validation set. OOB score ≈ cross-validation score, for free.

## Examples

1. **Customer churn prediction**: 10K customers, 50 features. Random forest achieves 85% AUC without extensive tuning. Feature importance reveals top predictors: login frequency, support tickets, payment delay.
2. **Fraud detection**: Highly imbalanced (0.1% fraud). Random Forest with class weighting handles imbalance naturally. Outperforms logistic regression because fraud patterns involve complex feature interactions (amount × merchant type × time of day × location).
3. **ECG classification**: 200 features from heart signal processing. Random Forest achieves 95% accuracy on arrhythmia detection. Each tree votes on the heart condition; the ensemble is robust to noise in individual signal leads.
