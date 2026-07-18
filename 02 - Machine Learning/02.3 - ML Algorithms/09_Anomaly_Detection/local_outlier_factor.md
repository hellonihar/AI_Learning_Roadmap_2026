# Local Outlier Factor (LOF)

An anomaly detection algorithm based on **local density deviation**. A point is anomalous if its density is significantly lower than its neighbors.

## How It Works

1. For each point, compute the distance to its k nearest neighbors
2. Compute the **local density** of the point (inverse of average distance to neighbors)
3. Compare the point's density to the average density of its neighbors
4. **LOF score**: ratio of neighbor density to point density

```
LOF ≈ 1    → Density is similar to neighbors (normal point)
LOF > 1    → Lower density than neighbors (anomalous)
LOF ≫ 1   → Much lower density (strong anomaly)
LOF < 1    → Higher density than neighbors (in a dense cluster)
```

## Key Hyperparameter: n_neighbors

| Few neighbors (5-10) | Many neighbors (20-50) |
|---|---|
| Detects local anomalies | Detects global anomalies |
| Sensitive to local noise | Smoother, may miss local outliers |
| Good for varying density data | Good for uniform density data |

## LOF vs Isolation Forest vs One-Class SVM

| Aspect | LOF | Isolation Forest | One-Class SVM |
|---|---|---|---|
| **Approach** | Local density comparison | Random isolation | Boundary around normal |
| **Local anomalies** | ✅ Excellent | Moderate | Struggles |
| **Varying density** | ✅ Handles well (normalizes by local density) | Moderate | ❌ Assumes uniform density |
| **Interpretation** | LOF score > 1 = anomaly | Anomaly score | Distance from boundary |

## When to Use LOF

- Data has **clusters of varying density**
- Anomalies are **local** (relative to their neighborhood, not globally extreme)
- Need to detect outliers that are "unusual for their context"

## Examples

1. **Credit card fraud (contextual)**: A $500 transaction is normal for a wealthy user, anomalous for a low-income user. LOF detects this — it compares each transaction to the user's local neighborhood (similar users), not the global distribution. Global methods might miss this "contextual fraud."
2. **Sports analytics**: Athlete performance stats. A basketball player with excellent three-point shooting but poor rebounding is "normal" for guards but "anomalous" for centers. LOF compares each player to their position group (local neighborhood), not all players.
3. **Retail inventory**: Sales patterns by store. A store in a tourist area has different "normal" sales than a suburban store. LOF compares each store to similar stores (by size, location type), flagging stores with sales patterns unusual for their peer group.
