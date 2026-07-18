# Improper Evaluation Pitfall

## The Mistake

Evaluating the model in the wrong way, leading to over-optimistic or misleading performance estimates.

## Common Forms

### 1. Evaluating on Training Data
The model has already seen the data — expect near-perfect scores. This tells you nothing about real-world performance.

**Example**: "Our neural network achieves 99% accuracy!" — on the same data it was trained on. In reality, it memorized the training set and fails on new data.

### 2. Data Leakage in Validation
Accidentally using future or target information during evaluation (covered in data leakage).

**Example**: Time-series data randomly shuffled before cross-validation. The model uses "future" data to predict the past, inflating scores.

### 3. Incorrect Temporal Split
For time-series: random split instead of chronological split.

**Example**: Stock prediction model trained on 2020-2024 data but also tested on 2020-2022 data (which leaked into training through random split). Real-world forward-testing performance is much worse.

### 4. Using Test Set Multiple Times
Tuning hyperparameters based on test set performance, then reporting the same test set as final evaluation. The test set is now contaminated.

**Example**: "We achieved 94% on our held-out test set!" — after 100 iterations of "check test score, tweak model, check again." Real performance on truly unseen data: 86%.

### 5. Selecting Model Based on a Noisy Metric
Running 1000 experiments, picking the best one by chance (multiple comparisons problem).

**Example**: Training 1000 random forest models with different random seeds. The best one has 92% accuracy purely by luck. The true performance of that model is 85%. Fix: use a separate test set for final evaluation, or adjust for multiple comparisons.

## Fixes

- Never evaluate on training data
- Use time-based split for temporal problems
- Touch test set exactly once
- Hold out a separate validation set for tuning, test set only for final report
- Use cross-validation for small datasets
- Report confidence intervals, not just point estimates
