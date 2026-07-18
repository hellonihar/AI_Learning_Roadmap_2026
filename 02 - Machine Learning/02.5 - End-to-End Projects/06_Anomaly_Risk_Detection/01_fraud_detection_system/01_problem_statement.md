# Fraud Detection System — Problem Statement

## Business Context
A digital payments platform processes 500K+ transactions daily. Fraud causes ~1.2% revenue leakage. Manual review of flagged transactions is expensive (~$4/review). Need an automated system that catches fraud while minimizing false positives.

## Problem Type
- **Primary**: Binary classification (fraud/legitimate) — highly imbalanced (<0.5% fraud rate)
- **Secondary**: Unsupervised anomaly detection for novel fraud patterns

## Success Metrics
- **Precision-Recall AUC** (primary, due to extreme imbalance)
- Recall @ 5% false positive rate
- Cost savings = (false negatives × avg fraud amount) - (false positives × review cost)
- Inference latency < 50ms per transaction
