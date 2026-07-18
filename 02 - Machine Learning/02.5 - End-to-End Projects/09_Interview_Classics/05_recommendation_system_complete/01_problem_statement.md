# Recommendation System — Problem Statement

## Business Context
A movie streaming platform wants to provide personalized recommendations to users. Collaborative filtering predicts user preferences based on historical ratings. The system must handle cold-start users and serve recommendations via a REST API.

## Problem Type
Collaborative filtering — predict user rating (1–5) for unseen items.

## Success Metrics
- **RMSE** < 0.95 on test ratings.
- **Cold-start handling**: recommend popular items for new users.
- **API latency** < 200 ms per request.

## Constraints
- Must handle cold-start users (no history) and cold-start items (no ratings).
- Must serve predictions via REST API.
- Must include A/B test framework for evaluating recommendation quality.
