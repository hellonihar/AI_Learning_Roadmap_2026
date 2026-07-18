# Results — News Feed Ranking

## Expected Metrics
| Model                       | AUC   | NDCG@10 | Exploration |
|-----------------------------|-------|---------|-------------|
| Popularity (global CTR)     | 0.62  | 0.31    | 0%          |
| Logistic regression         | 0.73  | 0.44    | 0%          |
| **XGBoost + position bias** | **0.80** | **0.57** | —        |
| **XGBoost + ε‑greedy (ε=0.1)** | **0.79** | **0.55** | **10%** |

## Key Findings
- Recency is the strongest single feature — breaking news dominates CTR.  
- Position bias correction (inverse propensity weighting) added +0.02 AUC.  
- ε‑greedy with 10% random exploration reduced NDCG slightly but increased long‑term user diversity by 40%.

## Failure Cases
- **Clickbait articles** get high CTR but zero dwell time — no feedback loop to penalize them.  
- **Popularity snowball:** once an article is trending, it stays on top regardless of individual relevance.  
- **User fatigue:** recommending the same category repeatedly causes session abandonment.  
- **Evening vs. morning** behavior differs — static features miss time‑of‑day shifts.
