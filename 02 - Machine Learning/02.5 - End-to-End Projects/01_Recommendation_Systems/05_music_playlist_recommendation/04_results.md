# Results — Music Playlist Recommendation

## Expected Metrics
| Model                    | Recall@20 | MRR   |
|--------------------------|-----------|-------|
| Popularity (most played) | 11.3%     | 0.07  |
| Item co‑occurrence (PPMI)| 24.7%     | 0.16  |
| **GRU4Rec (GRU‑100)**    | **36.2%** | **0.27** |
| + artist bias correction | 34.8%     | 0.25  |

## Key Findings
- Session context (last 20 tracks is optimal) outperforms global popularity by 3×.  
- GRU4Rec captures short‑term dynamics (e.g., genre shifts mid‑playlist).  
- Artist bias correction hurt recall slightly but improved diversity and skip rate.  
- Negative sampling with 100 negatives per positive was critical for training stability.

## Failure Cases
- **Very short sessions** (1–3 tracks) have too little signal — fall back to artist or genre similarity.  
- **Live / DJ mixes** break the sequential assumption (tracks are mashed together).  
- **Long‑tail tracks** that appear in < 5 playlists are hard to predict accurately.  
- **Playlist reordering:** the model assumes strict order, but many users shuffle.
