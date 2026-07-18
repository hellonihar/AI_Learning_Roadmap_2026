# Unsupervised Learning

Model finds patterns in **unlabeled** data. No "right answer" given.

## Clustering

Group similar data points together.

**Examples:**
1. **Customer segmentation**: E-commerce clusters users by purchase history → "budget shoppers", "luxury buyers", "deal hunters" — target marketing campaigns per group
2. **Document clustering**: News articles grouped by topic → sports, politics, tech — without any human labeling
3. **Image compression**: Reduce color palette by clustering similar pixel colors, replacing each with its cluster centroid (k-means compression)

## Density Estimation

Estimate the probability distribution of data.

**Examples:**
1. **Anomaly detection**: Find regions of low density in network traffic — a connection with very low probability under the learned distribution is flagged as an attack
2. **Generative modeling**: Learn the distribution of handwritten digits, then sample new digit images from that distribution
3. **Outlier removal**: In sensor data, fit a density model; remove readings with < 1% probability before training a downstream model

## Common Algorithms

| Algorithm | Type | Use Case |
|---|---|---|
| k-Means | Clustering | Spherical clusters, fast |
| DBSCAN | Clustering | Arbitrary shapes, with noise |
| Hierarchical (agglomerative) | Clustering | Dendrogram visualization |
| Gaussian Mixture Model | Clustering + Density | Soft assignments |
| PCA | Dimensionality reduction | Compression, visualization |
| t-SNE / UMAP | Dimensionality reduction | Visualization (2D/3D) |
