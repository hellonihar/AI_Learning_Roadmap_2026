# Bernoulli Naive Bayes

Variant of Naive Bayes for **binary features** — each feature is 1 (present) or 0 (absent).

## How It Works

Instead of counting occurrences, BernoulliNB checks **whether** a feature appears at all. Word presence matters, but frequency within a document does not.

```
P(doc | class) ∝ Π P(word present | class) × Π (1 — P(word absent | class))
```

## When to Use

- Binary feature vectors
- Short texts where word frequency within a document is less informative than presence/absence
- Features like "clicked" (0/1), "purchased" (0/1), or "visited" (0/1)

## Comparison with MultinomialNB

| | MultinomialNB | BernoulliNB |
|---|---|---|
| Features | Word counts | Word presence (0/1) |
| Sensitive to | Word frequency | Word occurrence |
| Best for | Long documents, full articles | Short texts, tweets, flags |
| Example | "free" appears 5 times vs 0 times | "free" appears (yes/no) |

## Examples

1. **Tweet sentiment**: Tweets are short (280 characters). Whether "amazing" appears matters more than how many times. BernoulliNB captures presence patterns better than count-based variants.
2. **Medical symptom checker**: Binary features — "fever (yes/no)", "cough (yes/no)", "fatigue (yes/no)". Each symptom is either present or absent. Frequency doesn't apply.
3. **Customer conversion prediction**: Binary features — "visited_pricing_page", "used_free_trial", "opened_email_campaign", "attended_webinar". Each is a flag. BernoulliNB predicts probability of conversion from behavioral flags.
