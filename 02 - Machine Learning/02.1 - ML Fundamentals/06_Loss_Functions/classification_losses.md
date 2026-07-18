# Classification Losses (Log Loss — Intuition)

## Log Loss (Binary Cross-Entropy)

The standard loss for binary classification. It measures how "surprised" the model should be.

```
log_loss = - ( y*log(p) + (1-y)*log(1-p) )
```

Where:
- `y` = actual label (0 or 1)
- `p` = predicted probability that y=1

### Intuition

- If the model predicts 95% probability for a correct label → log loss is small (low surprise)
- If the model predicts 5% probability for the actual correct label → log loss is very large (high surprise, heavily penalized)
- Going from 50% to 90% confidence on a correct prediction reduces log loss significantly
- Squeezing 95% → 99% helps much less — diminishing returns

## Why Models Minimize Loss, Not Accuracy

Accuracy is **non-differentiable** — it's a step function with flat gradients everywhere. You can't do gradient descent on accuracy.

Loss is **smooth and differentiable** — it gives a gradient signal that tells the model "move this way, even if you're already correct."

**Example**: Two predictions:
- Model A: p=0.51 for a positive case. At threshold 0.5, this is "correct" (accuracy += 1).
- Model B: p=0.99 for the same case. Also correct.

Both get the same accuracy score, but Model B has much lower loss. The loss function will push Model A toward Model B even though accuracy can't tell the difference.

## Examples

1. **Spam classifier**: Model predicts 90% spam for a spam email → log loss = -log(0.9) = 0.105. Model predicts 51% spam → log loss = -log(0.51) = 0.67. Both are correct predictions, but the 90% confidence model is much better (lower loss). Training optimizes for confident correct predictions.
2. **Medical test**: For a patient who actually has cancer, a model predicting 40% probability gets crushed by log loss (high penalty). The model will focus on pushing cancer probabilities above 50% for true positives. Accuracy just counts correct/final labels.
3. **Customer churn prediction**: A model predicting all customers at 49.9% churn probability is theoretically "accurate" (everyone predicted "not churn" at 0.5 threshold) but provides no useful signal. Log loss would be terrible because it's never confident. A model with some at 10%, some at 90% would have much better log loss.
