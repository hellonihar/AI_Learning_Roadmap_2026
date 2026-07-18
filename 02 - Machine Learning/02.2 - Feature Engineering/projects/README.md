# Feature Engineering — Projects

## Project 1: House Price Feature Engineering

Using the Ames Housing dataset (or similar), apply the full feature engineering workflow.

**Tasks:**
1. Understand each feature's type (numerical, categorical, ordinal, datetime)
2. Handle missing values — try mean imputation, flag + impute, and compare
3. Encode categorical variables — one-hot (low cardinality), ordinal (quality ratings)
4. Create features: `price_per_sqft`, `house_age`, `total_sf = 1st fl SF + 2nd fl SF + basement SF`, `has_remodeled`, `quality_score = overall_qual × kitchen_qual`
5. Scale numerical features (StandardScaler)
6. Compare model performance with raw vs engineered features using a simple linear regression
7. Check: how much did RMSE improve after feature engineering?

**Skills tested**: All 13 sub-sections — understanding data, missing values, encoding, scaling, creation, evaluation.

---

## Project 2: Fraud Detection Feature Creation

Given raw transaction logs (timestamp, amount, merchant, user_id, location), engineer features from scratch.

**Tasks:**
1. Create **temporal features**: hour_of_day, is_weekend, days_since_last_tx
2. Create **user aggregations**: avg_tx_amount_30d, tx_count_24h, unique_merchants_7d
3. Create **ratios**: tx_amount / user_avg_amount, tx_amount / user_avg_merchant_amount
4. Create **interaction**: is_night × is_high_amount
5. Create **rolling features**: avg_amount_last_3_tx, count_same_merchant_last_hour
6. Train a model on raw features only, then on engineered features. Compare AUC.
7. Check for leakage: did any feature use information from the future?

**Skills tested**: Temporal features, aggregations, interactions, rolling windows, leakage awareness.

---

## Project 3: The "Compare Approaches" Experiment

Take a dataset with mixed feature types and compare three feature engineering strategies.

**Dataset**: Any Kaggle tabular dataset (e.g., Titanic, Walmart Sales, Customer Churn).

**Strategy A — Minimal**: Impute mean, drop categoricals, no new features
**Strategy B — Basic**: Impute median, one-hot encode low-cardinality, scale numericals
**Strategy C — Full**: Domain-driven features, interactions, target encoding, log transforms, binning

**Compare**:
- Model performance (RMSE, accuracy, AUC — depending on problem)
- Number of features in each strategy
- Training time
- Interpretability (can you explain the top 3 features in each?)

**Skills tested**: End-to-end pipeline design, trade-off awareness, evaluation.

---

## Submission Checklist

For each project, document:
1. What raw data you started with
2. What features you created (and why)
3. What transformations you applied
4. How you handled missing values and outliers
5. How you avoided leakage (train-test split timing)
6. Performance comparison: before vs after feature engineering
