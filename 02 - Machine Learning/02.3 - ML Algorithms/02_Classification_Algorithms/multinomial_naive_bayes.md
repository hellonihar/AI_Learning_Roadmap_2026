# Multinomial Naive Bayes

Variant of Naive Bayes for **discrete count data** — most commonly word counts in text classification.

## How It Works

Features represent counts (e.g., how many times each word appears in a document). The likelihood P(feature | class) is modeled with a multinomial distribution.

```
P(doc | class) ∝ Π P(wordᵢ | class)^count(wordᵢ)
```

## Key Difference from GaussianNB

| | GaussianNB | MultinomialNB |
|---|---|---|
| Feature type | Continuous | Counts / frequencies |
| Distribution | Normal | Multinomial |
| Best for | Sensor data, measurements | Text data, integer counts |
| Decision boundary | Quadratic | Linear (in log space) |

## Practical Tips

- Use with **TF-IDF** or **count vectors** (from CountVectorizer)
- Always use **Laplace smoothing** (α ≥ 1) to handle words not seen in training
- Works best with **balanced classes** — for imbalanced data, adjust class prior

## Examples

1. **News classification**: 50K articles across 5 categories, represented as word count vectors. MultinomialNB trains in seconds and achieves ~90% accuracy — hard to beat for topic classification.
2. **Language detection**: Character n-gram counts. "th", "ing", "the" → English. "der", "die", "und" → German. Fast enough to detect language in real-time for routing customer support tickets.
3. **Review rating prediction**: Counts of positive words ("excellent", "amazing") vs negative words ("terrible", "awful") predict 1-5 star ratings. Surprisingly competitive with deep learning for this task.
