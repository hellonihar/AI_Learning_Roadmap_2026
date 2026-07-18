# Handwritten Digit Recognition — Deployment Notes

## Inference Latency

| Step | Time per sample |
|------|----------------|
| HOG extraction | ~1.5 ms |
| SVC predict | ~2.0 ms |
| **Total** | **~3.5 ms** |

Well under the 10 ms target for real-time processing.

## Model Size

| Model | Disk Size |
|-------|-----------|
| HOG + SVC (support vectors ~2k) | ~4 MB |
| RF 200 trees | ~85 MB |

Prefer the SVM model for deployment due to smaller footprint.

## Retraining Strategy

- Retrain quarterly on new handwriting samples from production
- Monitor accuracy per digit — if a specific digit drifts, collect augmented samples for that class
- Store the top-10 worst misclassifications each week for manual review

## Serving

- Export the HOG+SVC pipeline via `joblib.dump` (~4 MB)
- Serve behind a REST API (Flask/FastAPI) accepting base64-encoded 28×28 images
- Batch prediction recommended for high throughput (&gt;100 images/second)
