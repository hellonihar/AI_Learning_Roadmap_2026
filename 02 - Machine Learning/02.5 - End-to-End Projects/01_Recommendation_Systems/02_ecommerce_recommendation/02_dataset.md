# Dataset — Amazon Review Data (Electronics)

## Source
[Jianmo Ni et al., UCSD](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) — Amazon Electronics 5‑core (users + items with ≥ 5 reviews).

## Size
| Split   | Rows    | Users | Items  |
|---------|---------|-------|--------|
| Full    | ~1.7M   | 192k  | 63k    |
| Train   | ~1.36M  | 192k  | 63k    |
| Test    | ~340k   | 192k  | 63k    |

## Features
- **Interaction:** `reviewerID`, `asin`, `overall` (1–5), `unixReviewTime`.  
- **Product metadata:** `title`, `price`, `brand`, `category` (hierarchical), `also_bought`.  
- **Derived:** user purchase count, product popularity, category embeddings.

## Target
`overall` rating (threshold: ≥ 4 = positive) or binary `purchased` flag.

## Known Challenges
- **Extreme sparsity:** ~0.001% density — most users buy ≤ 5 items.  
- **Popularity bias:** 80% of interactions involve 20% of products.  
- **Missing metadata:** ~30% of products lack a price or brand.  
- **Shilling / fake reviews:** outliers can skew collaborative signals.
