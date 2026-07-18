# Deployment Notes — Movie Recommendation

## Inference Latency
- SVD prediction: ~0.01 ms per (user, item) pair.  
- For top-10 scoring across 10k items: ~100 ms.  
- Use precomputed user-factor vectors + ANN (e.g., faiss) to reduce to < 5 ms.

## Retraining Frequency
- Full retrain: daily (off-peak) with new ratings from the past 24h.  
- Incremental updates via `SVD.sgd` partial fit possible but not natively supported in `surprise` — use `implicit` or `funk-svd` for online updates.

## API Design
```
GET /recommend?user_id={id}&k=10
Response: {"user_id": 42, "recommendations": [{"movie_id": 127, "score": 4.8}, ...]}
```
- Batch endpoint: `POST /recommend/batch` with list of user IDs.  
- Fallback: return top-N popular items when user is unknown.

## Monitoring
- Track RMSE drift on streaming ratings (alert if > 1.0).  
- Log coverage (% of movies ever recommended) — should stay above 40%.  
- A/B test offline metrics vs. online CTR / watch-time lift.
