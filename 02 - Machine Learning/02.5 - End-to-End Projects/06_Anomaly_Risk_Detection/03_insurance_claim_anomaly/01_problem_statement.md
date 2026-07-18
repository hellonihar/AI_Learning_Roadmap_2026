# Insurance Claim Anomaly Detection — Problem Statement

## Business Context
A health insurance provider processes 200K claims/month. ~3-5% are fraudulent or erroneous (upcoding, phantom charges, duplicate billing). Manual audit costs $50/claim and only covers 2% of claims. Goal: flag the top 5% of suspicious claims for review.

## Problem Type
- **Primary**: Unsupervised anomaly detection (fraud labels are rare and unreliable)
- **Secondary**: Reconstruction-based anomaly scoring

## Success Metrics
- **Reconstruction Error AUC** — how well anomaly scores separate known fraud from clean claims
- Precision @ Top 5% flagged (review team capacity constraint)
- Coverage: % of confirmed fraud caught in the top 5%
- Audit cost reduction vs. random sampling
