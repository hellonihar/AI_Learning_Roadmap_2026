# How Bias-Variance Shows Up in Practice

## Signs of High Bias (Underfitting)

**Signals:**
- Training error is high and close to validation error
- Increasing model complexity helps
- The model oversimplifies — treats distinct patterns the same

**Examples:**
1. **Linear regression on housing**: Training MAE = $80K, Validation MAE = $82K. Both are high. Adding interaction features (sqft × bedrooms) drops both to $55K. The original model was too simple.
2. **Spam filter using word count only**: Training accuracy = 72%, Validation = 71%. Adding email header features and sender reputation brings both to 94%. The original model couldn't distinguish spam patterns.
3. **Recommendation system predicting average rating**: Training MSE = 1.5, Validation MSE = 1.5. Adding user-specific and item-specific biases drops both to 0.9. The "average" model ignores all personalization.

## Signs of High Variance (Overfitting)

**Signals:**
- Large gap between training error (low) and validation error (high)
- Model changes dramatically with small data changes
- Predictions are unstable — small input changes cause large output changes

**Examples:**
1. **Deep network on small dataset**: Training accuracy = 99.9%, Validation accuracy = 72%. Model memorized 10K training images but fails on 2.5K new ones. Applying dropout and data augmentation closes the gap to 85%/82%.
2. **Polynomial regression with degree 20**: Training MAE = $2K, Validation MAE = $200K. The wiggly polynomial has extreme slopes between training points. Ridge regression (L2 penalty) smooths it: Training MAE = $15K, Validation MAE = $18K.
3. **Decision tree without pruning**: Training accuracy = 100% (leaf per training sample), Validation accuracy = 65%. Pruning to depth 5: Training = 82%, Validation = 79%.

## Why This Tradeoff Never Disappears

No free lunch: As you increase model complexity, bias decreases but variance increases.

- Simple models won't capture complex patterns (high bias)
- Complex models will capture noise (high variance)
- Having infinite data helps but doesn't eliminate the tradeoff
- Regularization is how we explicitly control the balance

## Practical Techniques

| Technique | Effect |
|---|---|
| More training data | Reduces variance (without increasing bias) |
| Simpler model | Reduces variance, increases bias |
| Regularization (L1/L2) | Reduces variance, slightly increases bias |
| Ensemble (random forest) | Reduces variance (averaging uncorrelated errors) |
| Early stopping | Reduces variance (stops before overfitting) |
| Feature selection | Reduces variance (fewer noise features to overfit) |
