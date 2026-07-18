# Dataset — MovieLens 100k

## Source
GroupLens — MovieLens 100k dataset.  
[Link](https://grouplens.org/datasets/movielens/100k/)

## Size
- 100,000 ratings from 943 users on 1,682 movies
- Rating range: 1–5 (integer)

## Files

| File | Rows | Description |
|------|------|-------------|
| u.data | 100,000 | user_id, item_id, rating, timestamp |
| u.user | 943 | user_id, age, gender, occupation, zip |
| u.item | 1,682 | movie_id, title, release_date, genres (19 binary) |
| u.genre | 19 | Genre list |

## Target
- `rating`: integer 1–5

## Known Challenges
- **Sparsity** — 100k ratings / (943 × 1682) = 6.3% density.
- **Cold-start** — new users/items have no ratings.
- **Implicit bias** — some users rate only 3–5, others use full range.
- **Temporal effects** — ratings from 1997–1998, no recent data.
