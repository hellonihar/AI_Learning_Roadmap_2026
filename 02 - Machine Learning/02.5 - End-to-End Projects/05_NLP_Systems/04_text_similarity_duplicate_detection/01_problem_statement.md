# Text Similarity / Duplicate Detection — Problem Statement

## Business Context
Q&A platforms (Quora, Stack Overflow), forums, and support desks need to detect duplicate questions/threads to merge them, reduce clutter, and direct users to existing answers.

## Problem Type
Binary classification (duplicate / not-duplicate) based on pairwise text similarity. Semi-supervised / metric learning.

## Success Metrics
- **ROC AUC** ≥ 0.82
- **Precision at 90% recall** ≥ 0.75
- **Inference time** < 5 ms per pair
