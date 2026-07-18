# Deployment Notes — E‑Commerce Recommendation

## Inference Latency
- Candidate generation: ~15 ms per user (iALS precomputed).  
- Feature computation + LGB scoring for 100 candidates: ~8 ms.  
- Total p99 latency: < 50 ms with feature cache.

## Retraining Frequency
- iALS factors: weekly (overnight) on all history.  
- LGB ranker: daily (user behavior shifts fast).  
- Pipeline: Spark ETL → feature store → model train → push to serving.

## API Design
```
POST /recommend
Body: {"user_id": "abc123", "context": {"page": "homepage"}}
Response: {"items": [{"asin": "B00X...", "score": 0.92}, ...], "strategy": "hybrid"}
```
- Fallback: top‑N per category when user history is empty.  
- Shadow mode: log ranking scores alongside served list for offline eval.

## Monitoring
- **Coverage:** % of catalog ever recommended — alert if < 15%.  
- **Freshness:** average age of recommended items (target < 30 days).  
- **Revenue proxy:** track add‑to‑cart rate per recommendation slot.  
- **Drift:** daily K‑L divergence of score distributions between train / serving.
