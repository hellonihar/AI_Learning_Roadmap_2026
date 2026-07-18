# Data Leakage (VERY Important)

Data leakage is when **information from outside the training set** influences the model during training, creating an unrealistic performance estimate.

In production, your model will fail — because the leaked information won't be available at inference time.

## Common Types of Leakage

### 1. Leakage from the Future (Time-Series)

Using future information to predict the past.

**Example**: Building a stock price prediction model. You create a feature "future_30d_return" (which requires knowing stock prices 30 days ahead) as a predictor. During training, this feature is perfectly correlated with the target → model looks amazing. In production, you can't know tomorrow's price today. The model fails completely.

**Fix**: Always split time-series data chronologically. Never use future data to create features.

### 2. Data Before Train/Test Split (Preprocessing Leakage)

Fitting scalers, imputing missing values, or encoding categories **before** splitting.

**Example**: You scale all features using mean/std computed on the entire dataset (including test set). The test set's distribution has subtly influenced the scaling of the training data. The model sees validation data that was partially normalized with test-set statistics → performance is inflated.

**Fix**: Split first, then fit preprocessing ONLY on the training set. Apply the same transformation to val/test (don't refit).

### 3. Duplicate or Near-Duplicate Samples

Identical or nearly identical samples appearing in both train and test sets.

**Example**: Training an image classifier on internet photos. The same cat photo (or near-copy with different filename) ends up in both training and test sets. The model "recognizes" the exact photo rather than generalizing to "cateness". Reported accuracy: 98%. Real accuracy on unseen cat photos: 72%.

**Fix**: Deduplicate before splitting. Use hashing or embedding similarity to find and remove near-duplicates.

### 4. Feature Leakage (Target Leakage)

A feature contains information that would not be available at prediction time.

**Example**: Training a model to predict "patient will be readmitted within 30 days." You include "number of medications prescribed during readmission" as a feature — this requires knowing the readmission happened, which is exactly what you're trying to predict.

**Fix**: For each feature, ask: "Would I have this information at the moment of prediction?" If no, remove it.

### 5. Validation Contamination in Hyperparameter Search

Using validation set performance to tune hyperparameters, then reporting final performance on the same validation set.

**Example**: You try 1000 combinations of hyperparameters, each evaluated on the validation set. The best combination happens to work well on this specific validation set by chance. Your reported performance is optimistic.

**Fix**: Either use cross-validation, or have a completely separate test set used exactly once at the end.

## The Cost of Leakage

1. **Medical AI startup**: Used a feature "procedure_code" that was entered after diagnosis. Model was "94% accurate" in retrospective data. In deployment, procedure codes weren't available at prediction time → actual accuracy: 52% (random).
2. **Credit scoring model**: Used "number of late payments in the last 12 months" as a feature — but this included months after the loan was granted. Model looked fantastic. In production for new applicants (no payment history yet), it was useless.
3. **Kaggle competition**: A team found that duplicate images from the same source appeared in both train and test sets. Their model achieved near-perfect scores by memorizing those exact images. Organizers detected this and disqualified them.

## Golden Rules

1. **Split first** — before ANY preprocessing
2. **Evaluate on unseen data** — only the final model touches the test set
3. **Ask "would I have this at prediction time?"** — for every feature
4. **Deduplicate** — check for near-duplicates across splits
5. **Suspect high accuracy** — if a model is "too good" on a real-world problem, check for leakage
