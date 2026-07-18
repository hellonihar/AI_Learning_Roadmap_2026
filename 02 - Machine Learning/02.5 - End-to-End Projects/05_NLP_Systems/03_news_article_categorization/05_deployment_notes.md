# News Article Categorisation — Deployment Notes

## Inference Latency

| Step | Time per article |
|------|----------------|
| TF-IDF transform | ~0.8 ms |
| Linear SVC predict | ~0.04 ms |
| **Total** | **~0.85 ms** |

Barely measurable — LinearSVC is a sparse dot product. 1M articles processed in ~14 seconds on a single core.

## Model Size

| Artifact | Size |
|----------|------|
| TF-IDF vectoriser (10k vocab) | ~8 MB |
| Linear SVC coefficients (20 × 10k) | ~1.6 MB |
| **Total** | **~10 MB** |

## Retraining Strategy

- Retrain weekly for news aggregators (new events bring new vocabulary)
- Monitor per-class accuracy decay — if `sci.space` drops, trending topics may have changed the vocabulary
- Human-in-the-loop: flag articles where max probability < 0.6 for manual labelling
- LDA topic model should be recomputed monthly for category drift

## Deployment

- Serialise pipeline with `joblib`; hot-reload from S3/GCS on startup
- REST API: `POST /classify` → `{"category": "sci.space", "confidence": 0.94, "topic_vector": [...]}`
- For streaming (Kafka), batch 100 articles at a time for throughput efficiency
- Memory: ~50 MB RSS for loaded model + vectoriser — runs comfortably on a t3.micro (1 GB RAM)
