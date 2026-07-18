# Hierarchical Clustering

Builds a **tree of clusters** (dendrogram) without specifying the number of clusters upfront.

## How It Works (Agglomerative)

1. Start with each point as its own cluster
2. Find the two **closest** clusters
3. Merge them
4. Repeat until all points are in one cluster

## Linkage Criteria

| Linkage | Distance Between Clusters | Behavior |
|---|---|---|
| **Single** | Minimum distance between any two points | Creates long, chained clusters. Sensitive to noise. |
| **Complete** | Maximum distance between any two points | Creates compact, spherical clusters. Sensitive to outliers. |
| **Average** | Average distance between all pairs | Balance between single and complete. Most common. |
| **Ward** | Minimizes within-cluster variance | Creates equal-sized clusters. Works well with Euclidean distance. |

## Dendrogram Interpretation

```
Height  |   ┌──┐
        |   │  └──────┐
        |   │         └─────┐
        | ┌─┘               └──┐
        | └────────────────────┘
        └──────────────────────────→ Points
```

Cut the dendrogram horizontally at a chosen height = choose the number of clusters.

## When to Use

- Unknown number of clusters (dendrogram helps decide)
- Small datasets (<5K points) — hierarchical is O(n³) worst case
- Need to understand cluster relationships at multiple granularities

## Examples

1. **Gene expression analysis**: 1,000 genes measured under different conditions. Hierarchical clustering groups similarly expressed genes. The dendrogram reveals biological pathways — genes in the same branch tend to participate in the same cellular processes.
2. **Customer segment hierarchy**: All customers → broad segments (high/low value) → sub-segments (frequent big spenders, occasional big spenders, frequent small spenders, etc.). Marketing strategies differ at each level.
3. **Document topic hierarchy**: 5K news articles → sports has sub-topics (football, basketball, cricket) → further sub-topics (IPL, Premier League). The dendrogram reveals the natural topic taxonomy.
