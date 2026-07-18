# Imbalanced Data Handling

When one class has significantly fewer samples than others, standard models ignore it — achieving "high accuracy" by predicting the majority class for everything.

## The Problem

| Scenario | Positive | Negative | Accuracy of "always negative" |
|---|---|---|---|
| Fraud detection | 0.1% | 99.9% | 99.9% |
| Cancer screening | 1% | 99% | 99% |
| Rare disease | 0.01% | 99.99% | 99.99% |

Accuracy is meaningless. The model learns nothing about the minority class.

## Oversampling

Duplicate samples from the minority class to balance the dataset.

**Random Oversampling**: Randomly copy minority samples until balance is achieved.

**Pros**: Simple, increases minority signal.
**Cons**: Overfitting (model memorizes exact copies). No new information.

**Example**: Fraud dataset with 1,000 fraud + 999,000 legitimate. Oversample fraud to 100,000 copies. The model now sees enough fraud examples — but it may memorize the 1,000 original patterns rather than learning general fraud indicators.

## Undersampling

Remove samples from the majority class to balance the dataset.

**Random Undersampling**: Randomly drop majority samples until balance is achieved.

**Pros**: Simple, reduces training size.
**Cons**: Discards potentially useful majority data.

**Example**: Same fraud dataset — undersample legitimate to 1,000. Now the model trains on only 2,000 total samples. It sees fraud clearly but loses the diversity of legitimate transaction patterns.

## SMOTE (Synthetic Minority Oversampling Technique)

Creates **synthetic** minority samples by interpolating between existing minority points. Not copies — new samples.

**How it works**: For a minority sample, find its k nearest minority neighbors. Pick a random neighbor. Create a new sample at a random point on the line between them.

```
Original: (x₁, y₁) and (x₂, y₂)
New sample: (x₁ + α × (x₂ - x₁), y₁ + α × (y₂ - y₁)), where α ~ Uniform(0,1)
```

**Pros**: Creates realistic new samples (not duplicates), reduces overfitting compared to random oversampling.
**Cons**: Can create unrealistic samples in high dimensions. Doesn't work well with categorical features.

**Example**: Handwritten digit recognition — minority digit "7" has 50 samples. SMOTE creates 450 synthetic "7"s by interpolating between pairs of real "7" images. The new images look like slightly different 7s (thicker/thinner strokes, slight rotations). The model sees richer variation than simple copying.

## Class Weighting

Assign higher **misclassification penalties** to minority classes. No data modification needed.

Most algorithms support `class_weight='balanced'` which automatically sets weight inversely proportional to class frequency.

```
weight_minority = total_samples / (n_classes × n_minority_samples)
weight_majority = total_samples / (n_classes × n_majority_samples)
```

**Example**: Fraud with 1:1000 ratio. Setting `class_weight='balanced'` makes fraud errors 1000× more costly than legitimate errors. The model will sacrifice overall accuracy to catch fraud — exactly what you want.

## Threshold Moving

Standard classification uses 0.5 as the decision threshold. For imbalanced data, this is rarely optimal.

- **If recall is critical** (catch all fraud): lower threshold (e.g., predict "fraud" if probability > 0.2)
- **If precision is critical** (avoid false alarms): raise threshold (e.g., predict "fraud" if probability > 0.8)

**How to find the optimal threshold**: Plot precision-recall curve for different thresholds. Choose based on business cost of false positives vs false negatives.

**Example**: Fraud detection — 0.5 threshold gives 40% recall (catches 40% of fraud). Lowering to 0.2 gives 85% recall but floods the fraud team with 10× more false alerts. The business decides: "We can investigate 500 alerts/day" → choose threshold that limits alerts to 500, maximizing recall within that constraint.

## Comparison Table

| Technique | Best For | Trade-off |
|---|---|---|
| Oversampling | Small datasets, balanced training sizes | Overfitting risk |
| Undersampling | Very large datasets | Loses majority information |
| SMOTE | Medium datasets, continuous features | Can create unrealistic samples |
| Class weighting | Any size, integrated into model | Need to check model support |
| Threshold moving | After training, fine-tuning | Doesn't change model — just decision policy |

## Examples

1. **Credit card fraud**: 0.01% fraud rate. Strategy: SMOTE + class weighting in XGBoost (`scale_pos_weight=100`). Threshold moving on precision-recall curve to balance fraud catch rate with false positive budget. Final model: catches 80% of fraud with a 0.1% false positive rate.
2. **Medical diagnosis**: 1 in 10,000 patients has a rare disease. Strategy: class weighting is simplest (doctors understand "this test is more sensitive"). Then threshold moving: lower threshold to 0.05 for screening, higher threshold to 0.9 for confirmation.
3. **Customer churn**: 5% churn rate. Strategy: oversampling + ensemble (Random Forest with balanced class weights). Churn is not extremely rare (5%), so oversampling the minority class sufficiently without overfitting.
