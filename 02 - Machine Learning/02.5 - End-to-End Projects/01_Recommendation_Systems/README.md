# Recommendation Systems — End-to-End Projects

This category covers five foundational recommendation scenarios, progressing from explicit feedback (ratings) to implicit feedback (clicks, sessions) and from memory-based CF to deep sequential models.

## Projects Overview

| # | Project | Data | Approach | Key Metric | Difficulty |
|---|---------|------|----------|------------|------------|
| 1 | **MovieLens SVD** | MovieLens 100k/1M — explicit ratings (1–5) | Collaborative filtering via SVD matrix factorization | RMSE ≤ 0.90, Precision@10 ≥ 35% | ⭐⭐ |
| 2 | **E‑Commerce Hybrid** | Amazon Electronics 1.7M reviews | iALS candidate generation → LightGBM LambdaRank | Recall@20 ≥ 25%, HitRate@20 ≥ 40% | ⭐⭐⭐ |
| 3 | **Online Course (Cold‑Start)** | Coursera-style course + enrollment data | Skill TF-IDF + two-tower embedding hybrid | Precision@5 ≥ 45% (cold-start ≥ 35%) | ⭐⭐⭐ |
| 4 | **News Feed CTR** | MIND dataset — 15M impressions | XGBoost CTR + ε-greedy explore/exploit | AUC ≥ 0.78, NDCG@10 ≥ 0.55 | ⭐⭐ |
| 5 | **Music Playlist (Session‑Based)** | Spotify Million Playlist — 66M track occurrences | GRU4Rec recurrent next-track prediction | Recall@20 ≥ 35%, MRR ≥ 0.25 | ⭐⭐⭐⭐ |

## Key Techniques
- **Rating prediction:** SVD, SVD++, bias-aware matrix factorization (`surprise`).
- **Implicit feedback:** Alternating Least Squares (iALS), Bayesian Personalized Ranking (BPR).
- **Hybrid systems:** Score blending, feature stacking with LambdaRank.
- **Cold-start:** Content similarity (TF-IDF, doc2vec), popularity fallback.
- **Session-based:** GRU4Rec, item co-occurrence (PPMI), negative sampling.
- **Explore vs. exploit:** ε-greedy, Thompson sampling, UCB.

## Libraries Used
`pandas`, `numpy`, `scikit-learn`, `surprise`, `implicit`, `lightgbm`, `xgboost`, `tensorflow`, `faiss`
