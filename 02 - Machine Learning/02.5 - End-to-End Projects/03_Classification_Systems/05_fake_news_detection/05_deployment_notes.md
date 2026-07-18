# Deployment Notes — Fake News Detection

## Inference Pipeline

```python
def score_article(text: str) -> dict:
    clean = re.sub(r"[^\w\s]", "", text.lower())
    vec = vectorizer.transform([clean])
    ling = np.array([[
        len(clean.split()),
        sum(1 for c in text if c.isupper()) / (len(clean.split()) + 1),
        text.count("!") / (len(re.findall(r"[.!?]", text)) + 1),
        TextBlob(text).sentiment.polarity,
        textstat.flesch_reading_ease(text)
    ]])
    X = hstack([vec, ling])
    proba = ensemble.predict_proba(X)[0, 1]
    return {"fake_prob": proba, "label": "fake" if proba >= 0.5 else "real"}
```

- Latency: ~120 ms per article (TF-IDF + 2 models).
- Model size: 45 MB (TF-IDF vectorizer + LR coefficients + RF trees).

## Retraining Strategy
- **Bi-weekly**: retrain on newly fact-checked articles from human reviewers.
- **Active learning**: sample articles where model confidence < 0.3 or > 0.7 for human review.
- **Adversarial updates**: periodically inject known adversarial rewrites into training set.

## Threshold & Policy
- **Score ≥ 0.85**: auto-flag with reduced distribution.
- **Score 0.50–0.85**: send to human fact-checker queue.
- **Score < 0.50**: no action.
- During elections, lower flag threshold to 0.70.

## Monitoring
- **Model**: weekly F1 on holdout set, PSI on text length and vocabulary.
- **Platform**: flag-to-fact-check ratio, average time-to-review.
- **Adversarial**: track how often flagged articles are rewritten and reposted.

## Infrastructure
- Deployed as gRPC service for low-latency ingest pipeline integration.
- Redis cache for articles already scored (duplicate detection).
- Audit log for all flagged articles with model version and threshold snapshot.
