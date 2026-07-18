# Product Image Classification — Deployment Notes

## Inference Latency

| Step | Time per sample |
|------|----------------|
| Color histogram | ~0.3 ms |
| GLCM features | ~0.8 ms |
| LR predict | ~0.02 ms |
| **Total** | **~1.1 ms** |

Well within the 50 ms budget, allowing batch processing on a single CPU core.

## Model Size

| Artifact | Size |
|----------|------|
| LR coefficients + scaler | ~15 KB |
| Feature extraction code | bundled in app |

Extremely lightweight — deployable on edge devices or serverless functions.

## Retraining Strategy

- When new product categories are added, retrain with updated label set
- Monitor per-class accuracy weekly; if a class drops below 75% F1, flag for feature re-engineering
- Cold-start for new categories: use confidence threshold — if max prob < 0.6, route to human review

## Deployment

- Package feature extraction + scaler + model into a single `joblib` bundle
- REST endpoint: `POST /classify` accepting image bytes
- For high throughput, pre-extract features at upload time and cache them
