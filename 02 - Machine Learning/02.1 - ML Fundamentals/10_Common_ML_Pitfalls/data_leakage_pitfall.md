# Data Leakage Pitfall

Already covered in detail in [Section 04](../04_Train_Validation_Test_Split/data_leakage.md).

## Quick Recap

- Preprocessing leakage: fitting scaler on full data before split
- Future leakage: using information not available at prediction time
- Duplicate contamination: same sample in train and test
- Target leakage: feature that reveals the target

## Real-World Examples

1. **Healthcare AI startup (2020)**: A "revolutionary" sepsis prediction model claimed 90% accuracy. The feature set included lab orders — but lab orders are typically placed after sepsis is suspected. In production (no lab orders yet for unseen patients), accuracy collapsed to 50%. The company had to redo their entire study.
2. **Retail demand forecasting**: A team used "promotion flag" as a feature — but promotions are planned months in advance. Their training data had perfect promotion → sales correlation. In production, the model couldn't predict demand for non-promotion periods because it over-relied on the flag. Fix: separate baseline demand learning from promotion uplift modeling.
3. **Credit risk model**: Included "number of late payments in the last 6 months" as a feature — but the target was "will default in the next 6 months." The same 6-month window contained both the feature and the target. Time-based split (train on 2023, test on 2024) revealed the model was worthless for predicting future defaults.
