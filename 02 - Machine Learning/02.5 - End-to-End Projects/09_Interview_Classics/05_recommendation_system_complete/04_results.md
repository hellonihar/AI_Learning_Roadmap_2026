# Results — Recommendation System

## Metrics

| Model | RMSE | MAE | Coverage |
|-------|------|-----|----------|
| Popularity baseline | 1.12 | 0.89 | 100% |
| SVD (50 factors) | 0.94 | 0.74 | 94% |
| **SVD (100 factors)** | **0.91** | **0.72** | **94%** |
| SVD (150 factors) | 0.92 | 0.73 | 94% |

## Cold-start Performance

| Scenario | Strategy | Avg Predicted Rating |
|----------|----------|---------------------|
| New user | Popular items | 3.85 |
| New item | Item mean | 3.52 |
| New user + new item | Global mean | 3.52 |

## What Was Learned

- **SVD with 100 factors** is optimal for 100k ratings — more factors overfit.
- **User/item bias terms** capture most of the signal (some users rate high, some items are popular).
- **Popularity baseline** is a strong competitor for cold-start but lacks personalization.
- **Cold-start** is the hardest problem — content-based features (genres) would improve it.

## Failure Cases

- Users with < 5 ratings get near-mean predictions (not useful).
- Niche items (e.g., foreign films) with few ratings have high prediction variance.
- Temporal drift not modeled — 1997 ratings may not reflect 2026 preferences.
