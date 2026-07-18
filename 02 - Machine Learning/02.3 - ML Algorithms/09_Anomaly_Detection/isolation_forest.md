# Isolation Forest

An anomaly detection algorithm based on the idea that **anomalies are few and different** — they are easier to "isolate" than normal points.

## How It Works

1. Build a random forest of **isolation trees**
2. For each tree, randomly select a feature and a random split value
3. Continue splitting until every point is isolated in its own leaf
4. **Anomalies**: Have shorter path lengths (easier to isolate — few splits needed)
5. **Normal points**: Have longer path lengths (need many splits to isolate)

```
Normal point: Needs 15 splits to isolate (dense region, hard to separate)
Anomaly:      Needs 3 splits to isolate (sparse region, easy to separate)
```

## Key Hyperparameters

| Parameter | Effect |
|---|---|
| `n_estimators` | Number of isolation trees (default 100) |
| `contamination` | Expected proportion of anomalies (0.1 = expect 10% anomalies) |
| `max_samples` | Samples per tree (smaller = faster, more focus on isolation) |

## Isolation Forest vs Other Anomaly Detectors

| | Isolation Forest | One-Class SVM | LOF |
|---|---|---|---|
| **Speed** | Fast (linear) | Slow (quadratic) | Moderate |
| **High dimensions** | Works well | Struggles | Struggles |
| **Interpretation** | Anomaly score | Distance from boundary | Local density ratio |
| **Contamination assumption** | Yes (needs estimate) | No | No |

## Examples

1. **Fraud detection**: 1M transactions, 200 features. Isolation Forest is fast enough to score each transaction in real-time (<1ms). Flagged transactions have unique feature combinations that normal transactions don't have — e.g., high amount + new merchant + unusual location + 3 AM.
2. **Network intrusion detection**: Millions of network packets per hour. Isolation Forest detects unusual traffic patterns — a packet with unusual port, unusual packet size, and unusual protocol combination is isolated quickly. Normal traffic patterns require many splits to separate.
3. **Manufacturing quality control**: 100 sensor readings per product. A defective product has unusual sensor patterns (e.g., vibration + temperature + pressure outside normal operating ranges). Isolation Forest flags them before visual inspection.
