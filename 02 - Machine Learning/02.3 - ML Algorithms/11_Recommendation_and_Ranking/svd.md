# Singular Value Decomposition (SVD)

A matrix factorization technique that decomposes any matrix into three matrices: **U, Σ, Vᵀ**

## Mathematical Intuition

```
R (m × n) = U (m × k) × Σ (k × k) × Vᵀ (k × n)
```

- **U**: User-to-latent-factor matrix (users × concepts)
- **Σ**: Diagonal matrix of singular values (concept importance — larger = more important)
- **Vᵀ**: Item-to-latent-factor matrix (concepts × items)

## How SVD Differs from Basic Matrix Factorization

| Aspect | Basic MF | SVD |
|---|---|---|
| **Factor relationship** | Arbitrary P, Q | Orthogonal U, V (columns are independent) |
| **Singular values** | No ordering | Ordered by importance (Σ) |
| **Interpretation** | Less structured | Components are ranked by variance explained |
| **Dimensionality reduction** | Choose k arbitrarily | Use Σ to explain variance, select top-k |

## Dimensionality Reduction via Truncated SVD

Keep only the top-k singular values and corresponding vectors:

```
R ≈ Uₖ × Σₖ × Vₖᵀ
```

This is equivalent to PCA for sparse matrices. The truncated SVD gives the best rank-k approximation of the original matrix.

## When to Use SVD

- **Dimensionality reduction** for sparse user-item matrices
- **Noise reduction** (drop small singular values = drop noise)
- As a building block for **collaborative filtering** (FunkSVD)
- **Feature extraction** from high-dimensional sparse data

## Examples

1. **Netflix-style recommendation (FunkSVD)**: Simon Funk's winning SVD implementation for the Netflix Prize decomposed 100M ratings into 20 factors. Each factor captured a latent concept (e.g., "age group appeal," "critical vs popcorn film"). The model achieved a 10% RMSE improvement over Netflix's own algorithm.
2. **Document retrieval (LSI/LSA)**: Decompose a term-document matrix using SVD. Terms and documents are projected into a latent "concept space." A search for "car" returns documents about "automobile" even if the word "car" doesn't appear — because they share a latent concept vector. This is Latent Semantic Indexing (LSI).
3. **Image compression**: A grayscale image is a matrix of pixel intensities. SVD on this matrix — keeping top 10% of singular values — reconstructs a compressed image. The lowest singular values contain imperceptible noise; dropping them saves storage with minimal quality loss (achieved 10:1 compression for testing).
