# Collaborative Filtering

"People like you also liked..." — recommends items based on the behavior of similar users.

## How It Works

Collaborative filtering uses only **user-item interaction data** (ratings, purchases, clicks) — no user profiles or item features needed.

### User-Based
1. Find users who have similar rating/behavior patterns to the target user
2. Recommend items that those similar users liked but the target hasn't seen

### Item-Based
1. Find items that are frequently interacted with together
2. Recommend items similar to ones the target user has already interacted with

Item-based is generally faster and more stable than user-based (item relationships change slower than user tastes).

## Memory-Based vs Model-Based

| Approach | How It Works | Example |
|---|---|---|
| **Memory-based** | Compute similarity directly on raw data (kNN on user or item vectors) | "Users who liked A also liked B" |
| **Model-based** | Learn latent factors from the interaction matrix | Matrix factorization (SVD), neural collaborative filtering |

## Cold Start Problem

Collaborative filtering fails for **new users** (no history) and **new items** (no interactions). Solutions:
- Recommend popular items as fallback
- Use content-based filtering as a warm-start
- Ask the user for initial preferences

## Examples

1. **Amazon "Customers who bought this also bought"**: Item-based collaborative filtering on purchase data. If users who buy a laptop also buy a laptop bag, the system shows the bag as a recommendation — without knowing anything about laptops or bags.
2. **Netflix "Because you watched"**: User-based collaborative filtering. If User A and User B both rated *Stranger Things* 5 stars and *Dark* 4 stars, and User A rates *The OA* 5 stars, Netflix recommends *The OA* to User B — even though Netflix doesn't know the content of any show.
3. **Spotify Discover Weekly**: Collaborative filtering on listening patterns. Users who listen to similar artists get each other's discoveries. The feature engineering is minimal — just the play counts.
