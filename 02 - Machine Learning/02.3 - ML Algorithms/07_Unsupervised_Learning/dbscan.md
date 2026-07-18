# DBSCAN

**Density-Based Spatial Clustering of Applications with Noise**. Finds clusters of arbitrary shape based on **density** — not distance to a centroid.

## How It Works

1. For each point, count how many points are within **ε (eps)** distance
2. If count ≥ **min_samples** → mark as **core point**
3. **Core points** within ε of each other form a cluster
4. Points within ε of a core point but not core themselves → **border points**
5. All other points → **noise** (not assigned to any cluster)

## Key Hyperparameters

| Parameter | Effect |
|---|---|
| **ε (eps)** | Maximum distance for two points to be considered neighbors. Small ε → many points labeled noise. Large ε → clusters merge. |
| **min_samples** | Minimum points to form a dense region. Higher = fewer noise points, smoother clusters. |

## How DBSCAN Differs from K-Means

| Aspect | K-Means | DBSCAN |
|---|---|---|
| Cluster shape | Spherical | Arbitrary shapes |
| Number of clusters | Must specify upfront | Automatic (determined by density) |
| Noise handling | Forces all points into clusters | Labels noise points (unclustered) |
| Outlier sensitivity | High | Low (outliers = noise) |
| Varying density | Poor | Can struggle (need tuned eps per density) |

## Examples

1. **Geographic clustering**: 50K GPS coordinates of crime incidents. DBSCAN finds high-crime neighborhoods (dense clusters) and ignores isolated incidents (noise). K-Means would split sparse suburban areas unnaturally.
2. **Astronomy (discovery of new stars)**: Astronomical observations — dense regions = existing star clusters, isolated points = potential new discoveries (noise). DBSCAN naturally separates signal from noise.
3. **Anomaly detection in manufacturing**: Sensor readings from a production line. Normal operation = dense clusters. Faulty sensors or anomalous readings = isolated noise points that DBSCAN labels automatically.
