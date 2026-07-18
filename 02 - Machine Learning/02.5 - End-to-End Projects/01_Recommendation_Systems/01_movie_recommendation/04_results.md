# Results — Movie Recommendation

## Expected Metrics
| Model     | RMSE  | Precision@10 |
|-----------|-------|--------------|
| Baseline (mean) | 1.13 | 18.2% |
| KNN (cosine)    | 0.98 | 27.4% |
| **SVD (100 factors)** | **0.91** | **36.8%** |

## Key Findings
- **SVD generalizes better** than memory-based KNN, especially for sparse users.  
- Increasing factors beyond 100 yields diminishing returns and risks overfitting (RMSE drops but Precision@10 plateaus).  
- Cold-start users (< 5 ratings) see RMSE > 1.2 — a separate popularity-based fallback helps.

## Failure Cases
- **New movies** with no ratings are unrecommendable unless side info (genres, cast) is incorporated.  
- **Blockbuster bias:** SVD over-recommends popular titles, reducing diversity.  
- **Temporal drift:** 5-year-old ratings misalign with current taste — time-weighted SVD (TimeSVD++) mitigates this.
