# Text Similarity / Duplicate Detection — Deployment Notes

## Inference Latency

| Step | Time per pair |
|------|--------------|
| TF-IDF cosine | ~0.5 ms |
| Jaccard + basic features | ~0.1 ms |
| WMD (full) | ~25 ms |
| LR predict | ~0.01 ms |
| **Total (no WMD)** | **~0.6 ms** |
| **Total (with WMD)** | **~25.6 ms** |

WMD is a bottleneck. In production, use the 5-feature model (no WMD) for near-instant inference.

## Model Size

| Artifact | Size |
|----------|------|
| TF-IDF vectoriser | ~8 MB |
| Word2Vec model (not serialised in pipeline) | ~1.6 GB |
| LR coefficients | ~0.2 KB |
| **Pipeline (no WMD)** | **~8 MB** |
| **Pipeline (with WMD)** | **~1.6 GB** |

## Retraining Strategy

- Retrain monthly on new question pairs from production
- Re-fit TF-IDF vocabulary quarterly (new slang, terms, products emerge)
- WMD model: word2vec is frozen — no retraining needed unless domain shifts significantly
- Monitor AUC weekly; if below 0.80, investigate vocabulary drift

## Deployment Considerations

- **Use the 5-feature model in production** — WMD adds cost with marginal value
- Pre-compute TF-IDF vectors for existing questions and store in a vector DB (FAISS) for fast candidate retrieval
- Architecture: candidate retrieval (cosine ANN) → pairwise classification (LR) → merge UI
- For high-traffic sites, batch nightly dedup (O(n²) is expensive — cluster then score within clusters)
- Memory: full pipeline < 100 MB without word2vec; ~1.7 GB with word2vec
