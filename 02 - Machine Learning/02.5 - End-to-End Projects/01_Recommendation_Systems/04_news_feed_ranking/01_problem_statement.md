# Problem Statement — News Feed Ranking (CTR Prediction)

## Business Context
A news aggregator app wants to maximize daily active users by ordering articles in each user's feed by relevance. Feedback is implicit (clicks, scrolls, dwell). The system must balance showing what the user likely wants to click (exploit) vs. surfacing new topics (explore).

## Problem Type
**Implicit CTR prediction + learning to rank.**  
- **Input:** User ID, article ID, user features (device, session count), article features (category, recency, source).  
- **Output:** Score per (user, article) — feed is sorted descending.  
- **Approach:** Gradient‑boosted CTR model + ε‑greedy exploration.

## Success Metrics
- **NDCG@k** (k = 10) — normalized discounted cumulative gain of clicked articles. Target ≥ 0.55.  
- **AUC** — area under ROC for click prediction. Target ≥ 0.78.  
- **Exploration rate** — % of impressions that are non‑greedy picks (target 5–10%).
