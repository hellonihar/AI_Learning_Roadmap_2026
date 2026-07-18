# ML Fundamentals — Projects

## Project 1: End-to-End ML Pipeline (Iris / Wine Dataset)

Apply everything from this section to a small, clean dataset.

**Tasks:**
1. Load the dataset (scikit-learn built-in: `load_iris()` or `load_wine()`)
2. Perform train/validation/test split (60/20/20)
3. Train 3 models: logistic regression, decision tree, random forest
4. Evaluate on validation set using accuracy, precision, recall, F1, confusion matrix
5. Check for overfitting (compare train vs validation metrics)
6. Choose the best model and evaluate once on the test set
7. Write a paragraph explaining your choice

**Skills tested**: split, model training, evaluation metrics, bias-variance, model selection

## Project 2: The "Bad ML" Exercise

Intentionally make every mistake and observe the damage.

**Tasks:**
1. Load a dataset. **Don't split** — fit scaler on the entire dataset, then split.
2. Train a deep decision tree (max_depth=50). Observe perfect training accuracy and poor validation accuracy.
3. Use accuracy as the metric on an imbalanced dataset (e.g., credit card fraud from Kaggle). Observe 99% accuracy with 0% recall.
4. Evaluate on training data. Observe inflated metrics.
5. Fix each mistake one by one and note how the metrics change.

**Skills tested**: data leakage, overfitting, metric selection, evaluation hygiene

## Project 3: Custom Metric Design

Design an evaluation strategy for a real-world scenario.

**Choose one:**
- **Medical screening**: 0.1% prevalence. Cost of false negative = 1000x cost of false positive. Design the evaluation (which metrics? what threshold? what validation strategy?).
- **Content moderation**: 10M posts/day, 5 moderators review 500 posts each. Model flags posts for review. Design the metric and threshold (precision@500? recall at reasonable false positive rate?).
- **Spam filter**: User tolerance for spam is 1-in-1000 emails. False positive (marking important email as spam) is unacceptable. Design evaluation strategy.

**Skills tested**: metric selection, business-aware evaluation, cost-sensitive learning
