# Network Intrusion Detection — Problem Statement

## Business Context
A mid-size enterprise needs to monitor internal network traffic for malicious activity. Signature-based IDS (e.g., Snort) misses zero-day attacks. Need an ML-based anomaly detector that flags unusual connection patterns without requiring known attack signatures.

## Problem Type
- **Primary**: Unsupervised anomaly detection (no attack labels for novel threats)
- **Secondary**: Binary classification for known attack types

## Success Metrics
- **ROC AUC** (primary — threshold-agnostic ranking quality)
- Detection rate (Recall) at 1% false positive rate
- Time to detect (for streaming applications)
- False positives per day (< 50/day at scale)
