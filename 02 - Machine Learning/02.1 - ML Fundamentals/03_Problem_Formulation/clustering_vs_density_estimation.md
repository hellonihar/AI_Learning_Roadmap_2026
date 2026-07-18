# Clustering vs Density Estimation

## Clustering

Group **data points** that are similar. Output: discrete cluster assignments.

**Examples:**
1. **Customer segmentation**: Cluster 10K e-commerce users by purchase history → "budget", "mid-range", "luxury" segments. Each user gets exactly one segment label.
2. **Image segmentation**: Group pixels in a medical scan into "tissue", "bone", "background" regions. Each pixel assigned to a cluster.
3. **Anomaly detection (cluster-based)**: Define normal clusters. A point far from any cluster centroid is an anomaly. Used in manufacturing defect detection.

## Density Estimation

Estimate the **probability distribution** that generated the data. Output: a probability density function.

**Examples:**
1. **Outlier detection (density-based)**: Fit a Gaussian distribution to credit card transaction amounts. A \$10K purchase when the user averages \$50 → very low probability → flag as fraud.
2. **Generative modeling**: Learn the density of handwritten digits → sample new digit images from that density (e.g., using a GAN or diffusion model).
3. **Data imputation**: Estimate the density of features with missing values → sample plausible values to fill in gaps, preserving the original distribution.

## Key Difference

| | Clustering | Density Estimation |
|---|---|---|
| **Output** | Discrete group labels | Probability density |
| **Goal** | Find natural groupings | Model the data distribution |
| **Can generate new data?** | No | Yes |
| **Handles noise?** | Some algorithms (DBSCAN) | Naturally (low-density = noise) |
| **Use case** | "Which group does this belong to?" | "How likely is this sample?" |
