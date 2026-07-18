# Sentiment Analysis — Deployment Notes

## Inference Latency

| Step | Time per review |
|------|----------------|
| Text cleaning | ~0.3 ms |
| TF-IDF transform | ~1.0 ms |
| LR predict | ~0.05 ms |
| **Total** | **~1.35 ms** |

Well under the 5 ms target. Batch processing of 1,000 reviews takes ~1.2 seconds.

## Model Size

| Artifact | Size |
|----------|------|
| TF-IDF vectoriser (vocab) | ~15 MB |
| LR coefficients | ~0.8 MB |
| **Total** | **~16 MB** |

MultinomialNB is similar (~16 MB) but with slightly lower accuracy.

## Retraining Strategy

- Retrain weekly if used for real-time social media monitoring (language evolves fast)
- Monitor accuracy on a human-labelled holdout set — if AUC dips below 0.92, trigger retraining
- For domain adaptation (e.g., reviews → tweets), warm-start by fitting TF-IDF on new domain then tuning LR

## Serving

- Serialise the full pipeline (cleaner → vectoriser → model) with `joblib`
- REST API: `POST /sentiment` with `{"text": "..."}` → `{"label": "pos/neg", "confidence": 0.95}`
- For high throughput, pre-vectorise and cache frequent phrases
- Language detection upstream required for multi-lingual pipelines
