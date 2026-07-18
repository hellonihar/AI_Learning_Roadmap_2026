# Deployment Notes — News Feed Ranking

## Inference Latency
- Feature computation: ~3 ms per candidate (pre‑computed aggregates).  
- XGBoost predict: ~0.5 ms per (user, article) pair.  
- Scoring 500 candidates for feed: < 30 ms — inline in the request path.

## Retraining Frequency
- **Online:** model updated every 4 hours via sliding window (last 72h of impressions).  
- **Full retrain:** weekly on all data to reset feature normalizations.  
- Streaming pipeline: Kafka → feature store → model server → feed API.

## API Design
```
POST /feed
Body: {"user_id": "u_789", "session_id": "s_abc", "page": 1}
Response: {"articles": [{"id": "n_001", "score": 0.87, "is_explore": False}, ...]}
```
- Exploration flag logged for A/B analysis.  
- Dedup endpoint ensures no article appears twice in a session.

## Monitoring
- **AUC drift:** monitor per‑hour AUC, alert if drops below 0.72.  
- **Explore fraction:** dashboard of % random items shown per user segment.  
- **Dwell time:** track median dwell for clicked articles (proxy for quality).  
- **Novelty:** % of articles never shown before to each user (target > 20% per week).
