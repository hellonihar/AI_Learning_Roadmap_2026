# Deployment Notes — Recommendation System

## Inference
- REST API with two endpoints:
  - `POST /recommend` — top-N items for user
  - `POST /predict-rating` — predicted rating for user-item pair
- Cold-start fallback: popular items list (cached, refreshed daily)
- Response time target: < 200 ms (SVD prediction is O(n_factors))

## Retraining
- Retrain SVD weekly with new ratings.
- Use `surprise.SVD.fit()` for incremental updates.
- Recalculate popular items list daily.

## Monitoring
- Track RMSE on recent ratings (7-day window).
- Monitor coverage (% of items ever recommended).
- Track diversity (average pairwise cosine similarity of recommendations).
- Alert if RMSE > 1.0 or coverage < 50%.

## A/B Test Design
- **Control**: Popularity-based recommendations
- **Treatment**: SVD-based recommendations
- **Primary metric**: Click-through rate (CTR)
- **Secondary metrics**: Average rating of consumed items, 7-day user retention
- **Duration**: 2 weeks
- **Sample size**: 5,000 users per arm (80% power for 5% CTR lift)
- **Analysis**: Two-sample t-test on CTR, bootstrap for confidence intervals
