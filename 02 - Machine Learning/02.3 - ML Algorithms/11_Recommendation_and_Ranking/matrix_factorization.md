# Matrix Factorization

Decomposes the **user-item interaction matrix** into lower-dimensional **latent factor matrices**.

## The Core Idea

Given a sparse matrix R (users × items) with known ratings/interactions:

```
R ≈ P × Qᵀ
```

Where:
- **P** (users × k): User latent factors (each user has a k-dimensional taste vector)
- **Q** (items × k): Item latent factors (each item has a k-dimensional property vector)
- **k**: Number of latent factors (hyperparameter, typically 10-200)

The predicted rating for user u on item i is the dot product: `R̂ᵤᵢ = Pᵤ · Qᵢ`

## Why It Works

Latent factors capture **hidden structure** in the data. For movies:

- Factor 1 might capture "comedy vs drama" (−5 to +5)
- Factor 2 might capture "blockbuster vs indie" (−5 to +5)
- Factor 3 might capture "family-friendly vs adult" (−5 to +5)

A user with vector [4, -2, 3] likes comedies, slightly prefers indie, and wants family-friendly content. A movie with similar vector is a good match.

## Key Concepts

### Regularization
Prevents overfitting by penalizing large factor values: `Loss = MSE + λ(||P||² + ||Q||²)`

### Bias Terms
Account for user and item tendencies:
- Some users rate everything higher (user bias)
- Some movies are universally better (item bias)
- `R̂ᵤᵢ = μ + bᵤ + bᵢ + Pᵤ · Qᵢ`

### Cold Start
Matrix factorization can't handle new users/items. Must hybridize with content-based or use a separate cold-start model.

## Examples

1. **Netflix Prize (2006-2009)**: The famous competition that popularized matrix factorization. Decomposing 100M ratings into latent factors (k=20-50) improved RMSE by 10% over the baseline. The discovered latent factors revealed interpretable movie genres and user tastes.
2. **Last.fm music recommendation**: Listen counts as implicit feedback (no explicit ratings). Matrix factorization on the play-count matrix discovers latent "music taste" factors. Users with similar factors share similar taste, even across different artists.
3. **E-commerce purchase prediction**: User × product binary matrix (purchased / not purchased). Factor decomposition reveals product affinities: users who buy diapers also tend to buy baby wipes and baby food (Factor = "new parent items"). Recommendations leverage these discovered patterns.
