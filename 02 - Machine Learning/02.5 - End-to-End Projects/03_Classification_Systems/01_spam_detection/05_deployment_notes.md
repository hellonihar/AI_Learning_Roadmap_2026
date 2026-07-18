# Deployment Notes — Spam Detection

## Inference Pipeline

```python
def predict_spam(text: str, threshold: float = 0.65) -> dict:
    clean = re.sub(r"[^\w\s@.]", "", text.lower())
    vec = vectorizer.transform([clean])
    proba = model.predict_proba(vec)[0, 1]
    return {"is_spam": bool(proba >= threshold), "confidence": proba}
```

- Latency: ~8 ms / message on a single CPU core (Intel Xeon 3.0 GHz).
- Model size: 1.2 MB (MultinomialNB + vectorizer).

## Retraining Strategy
- **Trigger**: weekly or when spam precision drops below 0.97 in production monitoring.
- **Data pipeline**: user "mark as spam" / "not spam" actions serve as weak labels.
- **Feedback loop delay**: labels arrive 24–72 hours after inference.

## Threshold Tuning in Production
- Start at threshold from validation set (0.65).
- Monitor daily precision/recall via logged predictions + delayed labels.
- Adjust ±0.05 if precision drifts; log every change for audit trail.

## Monitoring
- **Alert on**: precision < 0.96, recall < 0.85, inference latency > 100 ms p99.
- **Drift detection**: compare token distributions (PSI) on a rolling 7-day window.

## Infrastructure
- Deploy as a REST endpoint (FastAPI) behind a load balancer.
- Stateless — scale horizontally with Redis-backed rate limiting.
