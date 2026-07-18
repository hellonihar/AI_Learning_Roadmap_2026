# Sentiment Analysis — Results

## Performance

| Model | Features | Accuracy | ROC AUC |
|-------|----------|----------|---------|
| LogisticRegression | Unigrams | 88.2% | 0.943 |
| LogisticRegression | Unigrams+Bigrams | **89.5%** | **0.956** |
| MultinomialNB | Unigrams | 85.0% | 0.921 |
| MultinomialNB | Unigrams+Bigrams | 86.1% | 0.932 |

## What Was Learned

- Bigrams add ~1.3% accuracy — critical for capturing negations ("not good" vs "good")
- Logistic Regression consistently beats MultinomialNB on this dataset
- Sublinear TF scaling (`sublinear_tf=True`) helps dampen overly frequent terms
- Top positive indicators: "excellent", "amazing", "brilliant", "must watch"
- Top negative indicators: "terrible", "boring", "worst", "waste of time"

## Failure Cases

- **Sarcasm** ("Great, another sequel nobody asked for") — misclassified as positive
- **Mixed reviews** — reviews that are 50% positive and 50% negative score near 0.5 and are often wrong
- **Short reviews** (< 10 words) lack signal for reliable classification
- Reviews containing "not bad" — bigrams help but some still slip through

## Edge Cases to Monitor

- Reviews with spelling errors lose n-gram matches
- Emoji-rich reviews (strip emoji → lose sentiment signal)
