# Results — E‑Commerce Recommendation

## Expected Metrics
| Model                     | Recall@20 | HitRate@20 |
|---------------------------|-----------|------------|
| Popularity baseline       | 8.2%      | 14.7%      |
| iALS (collaborative only) | 19.5%     | 33.1%      |
| **Hybrid (iALS + LGB)**   | **26.8%** | **43.6%**  |

## Key Findings
- Content features (price, category match, brand) add +7pp recall over pure CF.  
- LambdaRank loss outperforms pointwise regression for top‑k ranking by ~3pp NDCG.  
- Candidate pool size of 100 strikes best tradeoff between recall and latency.

## Failure Cases
- **New products** with few interactions fall back to category‑only matching, often irrelevant.  
- **Price‑sensitive users** ignored — model doesn't capture budget constraints.  
- **Seasonal trends** (e.g., holiday spikes) are missed without time‑aware features.  
- **Data leakage:** using `also_bought` from the same session inflates offline metrics.
