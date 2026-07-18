# Overconfidence in Incorrect Code

AI models present wrong answers with high confidence. They don't "know" when they're wrong.

## The Problem

AI never says "I'm not sure" unless explicitly prompted. It will generate plausible-sounding incorrect code with the same tone as correct code.

**Example:**

```
User: "Write a function to compute the F1 score manually"

AI:
def f1_score(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == p and t == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != p and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t != p and t == 1)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)
```

This looks correct. But what if `tp + fp == 0`? Division by zero. AI doesn't flag this unless asked.

## Why It's Dangerous

1. **Looks correct** — syntax is valid, structure is right
2. **Works on happy path** — passes basic tests
3. **Fails on edge cases** — silently wrong for boundary conditions
4. **No confidence indicator** — AI can't say "this might be wrong for edge case X"

## How to Protect Yourself

### 1. Ask for edge cases explicitly
```
"What could go wrong with this code? List all potential failure modes."
```

### 2. Ask for confidences
```
"How confident are you about this solution? 1-10. If not 10/10, what's uncertain?"
```

### 3. Test with known inputs
```python
assert f1_score([1, 0, 1], [1, 0, 1]) == 1.0
assert f1_score([1, 1, 1], [0, 0, 0]) == 0.0
```

### 4. Ask the AI to prove it
```
"Prove this works by walking through the logic with input X."
```
