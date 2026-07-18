# Content-Based Filtering

Recommends items **similar to what the user already liked**, based on item features — not other users.

## How It Works

1. Build a **profile** of each item (features: genre, director, keywords for movies; category, price, brand for products)
2. Build a **user profile** by aggregating features of items the user has liked in the past
3. Recommend items whose features are most similar to the user's profile

## Similarity Metrics

| Metric | Best For |
|---|---|
| **Cosine similarity** | Text features, TF-IDF vectors — magnitude is less important than direction |
| **Euclidean distance** | Numerical features on the same scale |
| **Pearson correlation** | Rating patterns (handles user bias) |

## Content-Based vs Collaborative Filtering

| Aspect | Content-Based | Collaborative |
|---|---|---|
| **Needs other users** | No | Yes |
| **Needs item features** | Yes | No |
| **Cold start (new item)** | ✅ Can recommend immediately | ❌ No interactions → no recommendations |
| **Cold start (new user)** | ✅ Needs initial preferences | ❌ No history → no recommendations |
| **Serendipity** | Low (similar to past) | High (discovers new tastes) |
| **Filter bubble** | Yes — keeps recommending similar items | Less — other users introduce variety |

## Hybrid Approaches

Most real-world systems combine both:

- **Weighted**: Score = α × content_score + (1-α) × collab_score
- **Cascaded**: Use content-based to warm-start, collaborative for refinement
- **Feature-augmented**: Use collaborative features as input to a content-based model

## Examples

1. **Movie recommendation for a niche user**: User loves obscure 1970s sci-fi. Collaborative filtering fails (not enough similar users). Content-based: find movies with same director, similar keywords ("space", "dystopian", "analog effects"), similar era. Makes relevant recommendations from content alone.
2. **Job recommendation**: Features = job title, skills required, industry, location, seniority. User profile = skills, past roles, industry. Recommend jobs with highest feature similarity to the user's successful past applications.
3. **News article recommendation**: Articles represented as TF-IDF vectors or embeddings. User reads 10 articles about AI and climate change. Content-based: find articles with similar topic vectors. No need for other readers' behavior — works instantly for breaking news.
