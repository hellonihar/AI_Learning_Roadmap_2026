# Ranking & Recommendation Problems

## Ranking

Given a set of items, order them by **relevance** to a query or user.

**Examples:**
1. **Search engine (Google)**: User searches "best Italian restaurant" → rank 10M web pages by relevance. The top result gets 30% of clicks, position #10 gets ~1%. Tiny ranking improvements translate to massive traffic changes.
2. **Job candidate screening**: Given 5,000 resumes for a "Data Scientist" role, rank by fit score. HR only reviews the top 50. ML ranking model learned from past hiring decisions.
3. **News feed**: For each user, rank 500 candidate news articles by predicted engagement. The top 10 appear in the feed. Must balance relevance, diversity, and freshness.

## Recommendation

Predict which items a user will **like or engage with**, and suggest them.

### Collaborative Filtering
"People like you also liked..." — uses user-item interaction patterns.

### Content-Based Filtering
"Because you liked this movie with X actors, here's another with similar actors" — uses item features.

### Hybrid
Combine both approaches (most real-world systems).

**Examples:**
1. **Netflix**: 260M users, thousands of titles. Collaborative filtering: users who both watched *Stranger Things* and rated it highly also tend to like *Dark*. Content-based: if you liked sci-fi with teen protagonists, here's more of that.
2. **Amazon**: "Customers who bought this also bought..." — collaborative filtering on purchase history. Also: "Based on items in your cart" → content-based product similarity.
3. **Spotify**: Discover Weekly — collaborative filtering on listening history + content-based audio feature analysis. Recommendations must balance: familiar artists (exploit) vs new discoveries (explore).

## Key Concepts

- **Cold start**: New user/item with no history → must use content-based or popularity-based fallback
- **Diversity vs relevance**: Showing 10 action movies might be relevant but boring — mix in a comedy
- **Exposure bias**: Model only knows about items it has shown → may never discover that a user likes a niche genre
