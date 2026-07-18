# Dataset — MovieLens 100k / 1M

## Source
[GroupLens Research](https://grouplens.org/datasets/movielens/) — MovieLens 100k (100,000 ratings from 943 users on 1,682 movies).

## Size
| Split   | Rows   | Users | Items |
|---------|--------|-------|-------|
| Full    | 100k   | 943   | 1,682 |
| Train   | ~80k   | 943   | 1,682 |
| Test    | ~20k   | 943   | 1,682 |

## Features
- `userId`, `movieId`, `rating` (1–5 integer), `timestamp`.  
- Movie metadata: title, genres (pipe-separated).  
- User demographics: age, gender, occupation, zip code (optional).

## Target
`rating` — explicit 1–5 star feedback.

## Known Challenges
- **Sparsity:** 100k ratings out of 1.6M possible pairs (~6.3% density).  
- **Cold-start users:** Users with < 5 ratings in train get poor predictions.  
- **Timestamp bias:** Ratings are not i.i.d. — newer movies appear later in time.  
- **No negative feedback:** Low ratings are explicit, not implicit negatives.
