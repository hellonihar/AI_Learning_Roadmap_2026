# Deployment Notes — Music Playlist Recommendation

## Inference Latency
- GRU inference: ~3 ms per session (on GPU) or ~15 ms (CPU int8 quantized).  
- Approximate nearest‑neighbor (faiss) on item embeddings: ~2 ms per query.  
- Total p99: < 30 ms — acceptable for "next up" auto‑play.

## Retraining Frequency
- GRU model: weekly (full MPD) + daily micro‑update with new sessions.  
- Embedding refresh: hourly via incremental co‑occurrence counts.  
- Training pipeline: TFX on GCP — 4h per epoch on 8 V100 GPUs.

## API Design
```
POST /next-track
Body: {"session_id": "s_xyz", "recent_tracks": ["tr_1", "tr_5", "tr_3"], "k": 5}
Response: {"tracks": [{"uri": "tr_9", "score": 0.82}, ...], "session_id": "s_xyz"}
```
- Cache top‑k for identical recent_tracks sequences (common patterns).  
- Shuffle the top‑5 with noise to avoid deterministic repeats.

## Monitoring
- **Recall@k per session length bucket** (short/medium/long).  
- **Skip rate** on recommended tracks — target < 30% in first 10 s.  
- **Coverage:** % of track catalog served — combat popularity lock‑in.  
- **Session stickiness:** increase in session length when recommendations are shown.
