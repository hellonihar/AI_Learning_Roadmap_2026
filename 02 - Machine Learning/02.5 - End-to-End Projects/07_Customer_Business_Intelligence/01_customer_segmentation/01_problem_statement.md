# Customer Segmentation — Problem Statement

## Business Context
An e-commerce company with 50K monthly active customers sends generic email campaigns (15% open rate, 2% click rate). Marketing team wants personalized targeting but has no customer segments. Need to group customers into meaningful segments based on purchase behavior.

## Problem Type
- **Primary**: Unsupervised clustering (K-Means)
- **Secondary**: Dimensionality reduction (PCA) for visualization

## Success Metrics
- Silhouette Score (internal coherence)
- Segment interpretability (business stakeholders must name each segment)
- Uplift in campaign metrics post-segmentation (A/B test)
- Coverage: % of customers assigned to a non-noise segment
