# Model Evaluation & Validation

Proper evaluation is how you know your model will work in the real world, not just on your training data.

## Cross-Validation Strategies

### k-Fold Cross-Validation

Split data into k folds. Train on k-1 folds, validate on the remaining fold. Repeat k times. Average the results.

```
Fold 1: [val] [train] [train] [train] [train]
Fold 2: [train] [val] [train] [train] [train]
Fold 3: [train] [train] [val] [train] [train]
Fold 4: [train] [train] [train] [val] [train]
Fold 5: [train] [train] [train] [train] [val]
```

- k=5 or k=10 are standard choices
- Lower variance estimate than a single train/val split
- Each sample is used for validation exactly once

### Stratified k-Fold

Preserves the **class distribution** in each fold. Essential for imbalanced classification.

**Example**: Dataset with 90% class A, 10% class B. Random k-Fold might produce a validation fold with 0% class B. Stratified ensures each fold has ~10% class B.

### Time-Series Validation

For temporal data, **never shuffle** — split chronologically to prevent future leakage.

**Expanding window**: Train on increasing windows, validate on the next time period.

```
Window 1: [train: Jan-Mar] → val: Apr
Window 2: [train: Jan-Jun] → val: Jul
Window 3: [train: Jan-Sep] → val: Oct
```

**Sliding window**: Fixed-size training window that moves forward.

```
Window 1: [train: Jan-Mar] → val: Apr
Window 2: [train: Feb-Apr] → val: May
Window 3: [train: Mar-May] → val: Jun
```

**Why**: Standard cross-validation shuffles time — a model would learn future patterns to predict the past, overestimating performance.

### Nested Cross-Validation

Two layers of cross-validation: the **inner loop** tunes hyperparameters, the **outer loop** estimates generalization performance.

```
Outer loop (5-fold, evaluates the model):
  Fold 1: [outer_train] [outer_val]
     Inner loop (3-fold, tunes hyperparams on outer_train only):
       Fold 1: [inner_train] [inner_val] → try 50 hyperparam combos
       Fold 2: [inner_train] [inner_val]
       Fold 3: [inner_train] [inner_val]
       Best hyperparams → train on outer_train → evaluate on outer_val
  Fold 2-5: repeat
```

**Why**: Without nested CV, tuning hyperparameters on the validation set and reporting that same validation score as "final performance" is optimistic. Nested CV gives an unbiased estimate of generalization.

**When to use**: Small datasets (<5K samples), comparing algorithms fairly, academic rigor.

## Comparison Table

| Strategy | Use Case | Bias | Variance |
|---|---|---|---|
| Hold-out (80/20) | Large data, quick iteration | Low (if split is representative) | High (one split) |
| k-Fold (k=5/10) | Standard evaluation | Low | Moderate |
| Stratified k-Fold | Imbalanced classification | Low | Moderate |
| Time-series (expanding) | Temporal data | Low (no future leakage) | Moderate |
| Time-series (sliding) | Temporal, concept drift | Low (adapts to recent) | Higher |
| Nested CV | Small data, fair comparison | Lowest | Highest (but honest) |

## Examples

1. **Medical model evaluation (nested CV)**: 500 patient records, 3 candidate algorithms. Nested CV: outer loop (5-fold) estimates each algorithm's real performance. Inner loop (3-fold) tunes hyperparameters independently per outer fold. Algorithm A reports 88% ± 2%, Algorithm B reports 87% ± 3%. Without nested CV, the best-tuned hyperparameters for each fold would give over-optimistic scores.
2. **Stock price prediction (time-series)**: Train on 2018-2022 data, validate on 2023, test on 2024. Standard k-Fold would let the model peek at 2024 patterns during training. Time-series CV with expanding windows ensures the model only sees past data. The 2024 test score is the true forward-testing performance.
3. **Fraud detection (stratified)**: 0.1% fraud rate. Without stratification, a validation fold might have zero fraud cases — the model appears "perfect" because it predicts "not fraud" for everything. Stratified k-Fold ensures each validation fold has the same fraud proportion as the full dataset.
