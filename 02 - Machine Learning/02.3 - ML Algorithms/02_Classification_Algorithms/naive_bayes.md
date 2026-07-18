# Naive Bayes

A probabilistic classifier based on **Bayes' Theorem** with a "naive" assumption: all features are independent given the class.

## Core Idea

```
P(class | features) ∝ P(class) × P(feature₁ | class) × P(feature₂ | class) × ...
```

Despite the naive independence assumption, it works surprisingly well for many real-world problems.

## The "Naive" Assumption

The model assumes features don't influence each other. This is almost always false (words in a document are clearly related), but the simplification makes computation feasible and often produces good results.

## When to Use

- Very high-dimensional data (text, bag of words)
- Fast training and inference needed
- Baseline for classification problems
- Moderate-to-large datasets

## Pros and Cons

| Pros | Cons |
|---|---|
| Extremely fast (linear time) | Independence assumption rarely holds |
| Works well with high dimensions | Can't learn feature interactions |
| Handles missing data naturally | Calibrated probabilities (not good for decision thresholding) |
| Small data OK | Zero-frequency problem (handled with smoothing) |

## Examples

1. **Spam detection**: P(spam | "free", "winner", "click") ∝ P(spam) × P("free"|spam) × P("winner"|spam) × P("click"|spam). Trained on 50K emails, achieves ~98% accuracy.
2. **Sentiment analysis**: Classify movie reviews as positive or negative based on word frequencies. "amazing", "loved" → positive. "terrible", "boring" → negative.
3. **Document categorization**: News articles → sports, politics, tech, entertainment. Fast enough to run on millions of documents in real-time.
