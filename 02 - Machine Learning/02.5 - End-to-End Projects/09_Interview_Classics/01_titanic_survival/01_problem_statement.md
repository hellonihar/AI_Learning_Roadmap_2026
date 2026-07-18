# Titanic Survival — Problem Statement

## Business Context
The sinking of the RMS Titanic in 1912 is a historical event. The goal is to predict which passengers survived based on attributes like class, sex, age, and family. This is the canonical ML interview dataset — it tests feature engineering from messy raw data.

## Problem Type
Binary classification — predict survival (1) or death (0).

## Success Metrics
- **Accuracy** ≥ 0.83 on held-out test set.
- **AUC-ROC** ≥ 0.87.
- Demonstrate feature engineering from Name, Cabin, Ticket.

## Constraints
- Must handle missing Age, Cabin, Embarked.
- No external data allowed.
- Must use cross-validation (not single split) for reliable estimate.
