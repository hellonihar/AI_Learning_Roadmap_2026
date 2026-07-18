# Principal Component Analysis (PCA)

A **dimensionality reduction** technique that finds new features (principal components) that capture the maximum variance in the data.

## How It Works

1. Find the direction of maximum variance in the data → **PC1**
2. Find the direction of maximum variance **orthogonal to PC1** → **PC2**
3. Continue for as many components as needed
4. Project the original data onto these components

## Variance Explained

Each component explains a percentage of the total variance. The first component explains the most, the second explains the second most, etc.

```
Component 1: 45% of variance
Component 2: 30% of variance
Component 3: 15% of variance
Component 4: 10% of variance
```

**Rule**: Keep enough components to explain 80-95% of variance. The "elbow" in the scree plot guides the choice.

## When to Use

- Visualization (reduce to 2-3 components for plotting)
- Noise reduction (drop low-variance components)
- Multicollinearity (components are uncorrelated)
- Preprocessing for models that assume independence

## What PCA Cannot Do

- It's **unsupervised** — doesn't use the target. Components may not align with what's predictive.
- Assumes **linear relationships** — can't capture non-linear structure (use Kernel PCA or t-SNE).
- **Interpretability suffers** — components are linear combinations of original features, not original features.

## Examples

1. **Face recognition (Eigenfaces)**: 10K face images, each 100×100 pixels (10K dimensions). PCA reduces to 200 components explaining 95% variance. Each "eigenface" component captures a facial pattern (lighting, orientation, facial structure). Classification on 200 features instead of 10K.
2. **Genomics**: 20K gene expressions, 500 samples. PCA reduces to 50 components for clustering or further analysis. The first component often captures the primary biological difference (e.g., cancer vs normal).
3. **Stock market analysis**: Price movements of 500 stocks. PCA reveals that 5-10 components explain most market variance. The first component is the "market" (all stocks move together), the second is "sector" (tech vs energy).
