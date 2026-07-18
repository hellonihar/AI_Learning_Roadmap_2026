# Problem Statement — Movie Recommendation (Collaborative Filtering + SVD)

## Business Context
A streaming platform wants to improve user retention by serving personalized movie recommendations. Users browse hundreds of titles but engagement drops when suggestions are irrelevant. The goal is to predict each user's rating for unseen movies and surface the top-N most likely to be enjoyed.

## Problem Type
**Rating prediction + Top-N ranking.**  
- **Input:** User ID, Movie ID, historical rating (1–5).  
- **Output:** Predicted rating for unseen (user, item) pairs.  
- **Approach:** Collaborative filtering via SVD matrix factorization, benchmarked against a baseline cosine-similarity KNN.

## Success Metrics
- **RMSE** (root mean squared error) on held-out ratings — target ≤ 0.90.  
- **Precision@k** (k = 10) — fraction of recommended movies the user actually rates highly (≥ 4) in the test set. Target ≥ 35%.  
- Trade-off: optimize for ranking quality, not just raw prediction accuracy.
