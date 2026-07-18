# Confusion Matrix Intuition

A confusion matrix shows **exactly where** your model makes mistakes.

## Binary Classification

```
              Actual Positive    Actual Negative
Predicted +   True Positive (TP)  False Positive (FP)  ← Type I error
Predicted -   False Negative (FN) True Negative (TN)   ← Type II error
```

**Examples:**
1. **Spam filter**: [[TP=90, FP=10], [FN=5, TN=895]]. 90 spam caught, 10 legitimate emails wrongly sent to spam (FP), 5 spam slipped through (FN), 895 legitimate delivered correctly.
2. **Cancer diagnosis**: [[TP=8, FP=2], [FN=1, TN=989]]. 8 cancers correctly identified, 2 false alarms (biopsies needed but no cancer), 1 cancer missed (dangerous), 989 healthy correctly cleared.

## Multi-Class Confusion Matrix

Each row = actual class, each column = predicted class.

```
       Predicted
       cat  dog  bird
Actual cat  45    3     2    ← 45 cats correct, 3 called dog, 2 called bird
       dog   4   42     4    ← 4 dogs called cat, 42 correct, 4 called bird
       bird  2    1    47    ← 2 birds called cat, 1 called dog, 47 correct
```

- **Diagonal**: correct predictions (134/150 = 89% accuracy)
- **Off-diagonal**: mistakes — shows which classes are confused
- Cat-bird confusion (2 each direction) suggests some visual similarity

## Why It Matters

Accuracy hides which errors you're making. The confusion matrix reveals:
- Which classes are hard (low diagonal values)
- Which classes get confused with each other
- Whether the model is biased toward a particular class

## Examples

1. **Medical imaging**: A confusion matrix for chest X-ray classification might show: pneumonia ↔ healthy: rarely confused (good), but pneumonia ↔ bronchitis: frequently confused (expected — they look similar). This tells doctors which predictions to verify carefully.
2. **Handwriting recognition**: Confusion matrix of digits might show: "1" misclassified as "7" occasionally (both are vertical lines). "8" rarely confused with any other digit (distinctive shape). This guides data collection — collect more "7" samples.
3. **Sentiment analysis**: Confusion matrix for positive/neutral/negative might show: neutral ↔ positive: common confusion (reviews with mixed sentiment). Negative ↔ positive: rare confusion (truly opposite sentiments are well-separated).
