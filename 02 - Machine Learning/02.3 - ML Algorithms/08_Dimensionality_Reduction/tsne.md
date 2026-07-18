# t-SNE

**t-distributed Stochastic Neighbor Embedding** — a non-linear dimensionality reduction technique designed for **visualization** (typically 2D or 3D).

## How It Works (Intuition)

1. In high-dimensional space, measure similarity between every pair of points (Gaussian distribution)
2. In low-dimensional space (2D), measure similarity between every pair of points (t-distribution)
3. Move the low-dimensional points to minimize the difference between the two similarity matrices

The result: points that are close in high dimensions stay close in 2D; points that are far apart stay far apart.

## Key Hyperparameter: Perplexity

**Perplexity** controls the balance between local and global structure.

| Low Perplexity (5-10) | High Perplexity (30-50) |
|---|---|
| Focuses on local neighborhoods | Captures more global structure |
| Small clusters may fragment | Clusters may merge |
| Noisier visualization | Smoother visualization |

**Rule**: Try perplexity 30 first. Adjust up/down based on whether the visualization looks meaningful.

## Critical Properties

- **Non-deterministic**: Running t-SNE twice gives different results
- **Distance preservation is local only**: Distances between distant clusters are not meaningful
- **Can't embed new points**: t-SNE is a visualization technique, not a general-purpose transformer
- **Computationally expensive**: O(n²) — slow for >10K points

## t-SNE vs PCA

| Aspect | PCA | t-SNE |
|---|---|---|
| **Output** | Transform (can project new data) | Visualization only |
| **Speed** | Fast | Slow |
| **Structure** | Linear, global | Non-linear, local |
| **Deterministic** | Yes | No |
| **Best for** | Preprocessing, feature extraction | Exploration, visualization |

## Examples

1. **MNIST digits visualization**: 70K digit images (784 pixels each). t-SNE projects to 2D — the 10 digits form 10 distinct clusters. You can visually see that "4" and "9" are close (similar shape), while "0" and "1" are far apart. PCA can't separate them this cleanly.
2. **Single-cell RNA sequencing**: 20K cells, 30K gene expressions. t-SNE reveals cell types that biologists previously didn't know existed — distinct clusters corresponding to different cell populations. This has led to biological discoveries.
3. **NLP document exploration**: 5K news articles as sentence embeddings. t-SNE visualizes the document landscape: politics cluster, sports cluster, etc. Outliers between clusters might be mixed-topic articles or misclassified documents.
