# Problem Statement — Music Playlist Recommendation (Session‑Based)

## Business Context
A music streaming service wants to suggest the next track in a user's active listening session. Unlike persistent user profiles, session context (current track, recent skips, time of day) is the primary signal. The system must predict the next track given the immediate listening history.

## Problem Type
**Session‑based next‑item prediction.**  
- **Input:** Sequence of track IDs within a session (playlist).  
- **Output:** Top‑N candidate tracks to play next.  
- **Approach:** GRU‑based recurrent model (session embeddings) + item co‑occurrence.

## Success Metrics
- **Recall@k** (k = 20) — next track appears in top‑20. Target ≥ 35%.  
- **MRR** (mean reciprocal rank) — rank of the correct next track. Target ≥ 0.25.  
- **Skip rate** — % of recommended tracks skipped within 10 seconds (proxy for relevance).
