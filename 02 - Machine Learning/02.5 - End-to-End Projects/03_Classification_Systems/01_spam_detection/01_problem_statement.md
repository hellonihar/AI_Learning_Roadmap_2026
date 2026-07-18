# Problem Statement — Spam Detection

## Business Context
Messaging platforms and email providers lose user trust when spam slips through. False positives (legitimate messages flagged as spam) are more damaging than false negatives — users must be able to rely on their inbox.

## Problem Type
Binary text classification: ham (0) vs. spam (1).

## Success Metrics
- **Precision (spam class) ≥ 0.98** — every spam flag must be justified.
- **Recall ≥ 0.90** — catch most spam without sacrificing precision.
- **AUC-ROC ≥ 0.99** — overall separability.
- F1 is secondary; precision is the hard constraint.

## Constraints
- Inference must complete in < 50 ms per message on a single CPU core.
- Model must be retrainable weekly with no manual labeling (user feedback loop).
