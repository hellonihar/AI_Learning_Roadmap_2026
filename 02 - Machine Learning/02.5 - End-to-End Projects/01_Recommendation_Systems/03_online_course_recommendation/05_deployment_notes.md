# Deployment Notes — Online Course Recommendation

## Inference Latency
- Skill‑TF‑IDF: ~2 ms per user (pre‑computed course matrix).  
- Two‑tower: ~5 ms for top‑k via faiss index on course embeddings.  
- Total p95: < 20 ms — fast enough for homepage personalization.

## Retraining Frequency
- Skill‑TF‑IDF matrix: weekly (new courses added).  
- Two‑tower model: monthly, or when new enrollment data exceeds 100k events.  
- Embedding refresh: continuous via incremental training scripts.

## API Design
```
POST /courses/recommend
Body: {"user_id": "u_42", "skills": ["python", "ml"], "k": 5}
Response: {"courses": [{"id": "c_101", "score": 0.91, "reason": "skill_match"}, ...]}
```
- Reason field helps product explain *why* a course was recommended.  
- Cache top‑k for anonymous users (most‑popular‑in‑skill‑category).

## Monitoring
- **Precision@5** per cohort (new vs. returning) — daily report.  
- **Skill‑coverage:** % of skill tags that produce at least one recommendation.  
- **Enrollment lift:** compare recommendation‑driven enrollments vs. organic browse.  
- **A/B test idle:** serve 10% of traffic with popularity baseline to measure incrementality.
