# Asking for Explanations and Refactors

AI is excellent at explaining code and suggesting improvements.

## Ask for Explanations

```
Prompt: "Explain this function line by line"
```
```python
def compute_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return {"precision": precision, "recall": recall}
```

**AI response:**
```
Line 2-4: Count true positives, false positives, false negatives by
iterating through paired true/predicted values.
Line 5: precision = TP / (TP + FP) — of predicted positives, how many correct
Line 6: recall = TP / (TP + FN) — of actual positives, how many found
```

## Ask for Refactors

```
Prompt: "Refactor this to use numpy for better performance"
```

```python
def compute_metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"precision": precision, "recall": recall}
```

## Useful Explanation Prompts

```
- "What does this line do?"  (highlight specific line)
- "What is the time complexity of this function?"
- "Why is this code slow?"
- "What would break if I remove this check?"
- "Simplify this for me"
```
