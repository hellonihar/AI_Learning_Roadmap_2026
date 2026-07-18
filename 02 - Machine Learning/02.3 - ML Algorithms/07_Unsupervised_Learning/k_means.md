# K-Means

Partitions data into **k clusters**, where each point belongs to the cluster with the nearest mean (centroid).

## How It Works

1. **Initialize**: Pick k random points as initial centroids
2. **Assign**: Assign each point to the nearest centroid
3. **Update**: Recompute centroids as the mean of all points in the cluster
4. **Repeat**: Steps 2-3 until centroids stop changing

## Choosing k

| Method | How It Works |
|---|---|
| **Elbow method** | Plot inertia (sum of squared distances) vs k. Look for the "elbow" where improvement slows. |
| **Silhouette score** | Measures how similar a point is to its own cluster vs other clusters. Higher = better. Range [-1, 1]. |
| **Gap statistic** | Compare inertia against random data. Choose k where the gap is largest. |

## Limitations

- Assumes **spherical clusters** of similar size — fails on elongated or irregular shapes
- Sensitive to **initial centroid placement** (run multiple times with different seeds)
- **Scales poorly** with large datasets (use Mini-Batch K-Means)
- All points are **assigned to exactly one cluster** (no soft assignment)
- Struggles with **varying density** — dense regions may be over-split

## Examples

1. **Customer segmentation**: 10K e-commerce customers clustered by purchase frequency, avg order value, recency. K=4: "budget shoppers", "loyal regulars", "big spenders", "dormant". Marketing tailors campaigns per segment.
2. **Image compression**: Reduce an image from 16M colors to 16 colors. Each pixel's RGB value is clustered (k=16). Each pixel is replaced by its cluster centroid. 16 colors → much smaller file, visually similar.
3. **Document grouping**: Cluster TF-IDF vectors of news articles. Each cluster becomes a topic. Centroids (top words near the center) reveal the theme: {"election", "vote", "campaign", "policy"} → politics cluster.
