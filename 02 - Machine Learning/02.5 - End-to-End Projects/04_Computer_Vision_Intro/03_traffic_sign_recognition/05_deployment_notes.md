# Traffic Sign Recognition — Deployment Notes

## Inference Latency

| Step | Time per sample |
|------|----------------|
| Resize + HOG + color | ~4 ms |
| PCA transform | ~0.2 ms |
| SVC predict | ~3 ms |
| **Total** | **~7.2 ms** |

Under 30 ms budget per frame. At 30 FPS video (33 ms/frame), the system can run near-real-time.

## Model Size

| Artifact | Size |
|----------|------|
| PCA (120 components) | ~0.6 MB |
| SVC (support vectors ~8k) | ~12 MB |
| Combined pipeline | **~13 MB** |

## Retraining Strategy

- Collect edge-case data from production (dusk, rain, snow) each month
- Fine-tune: retrain only the SVM on the expanded dataset (PCA is fixed, no need to recompute)
- If accuracy on speed-limit signs drops below 85%, add a dedicated digit-CNN (exception to the no-DL rule in practice)

## Deployment Considerations

- **Region-specific models**: German signs differ from US signs — maintain separate pipelines per market
- **Confidence gating**: if probability < 0.7, fall back to temporal smoothing across 3 consecutive frames
- **Hardware**: runs on a single ARM Cortex-A72 core at ~10 FPS; use NEON SIMD for ~3× speedup on HOG
