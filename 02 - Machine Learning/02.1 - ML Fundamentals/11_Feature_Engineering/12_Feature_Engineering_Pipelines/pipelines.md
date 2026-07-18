# Feature Engineering Pipelines

## Why Pipelines Matter

Feature engineering involves multiple steps applied in sequence. Doing them manually is error-prone, unreproducible, and — worst of all — easy to leak data.

A **pipeline** chains all transformations into a single reproducible object. Data flows through each step automatically.

## Train-Test Consistency

The single most important rule: **Apply the exact same transformations to train and test data.**

Without a pipeline, it's easy to accidentally:
- Compute mean income on training data, but accidentally use the full-dataset mean for test data
- Create a new feature for test data that uses a different logic than training
- Drop different columns in train vs test

A pipeline guarantees consistency — you `fit` once on training data and `transform` both train and test with the same learned parameters.

## Fit vs Transform Distinction

Every transformation has two phases:

| Phase | What Happens | Example |
|---|---|---|
| **fit** | Learn parameters from training data | Compute mean income, min/max of price, encoder mapping |
| **transform** | Apply learned parameters to data | Subtract training mean, scale by training range, map categories |

- **fit** is called **only on training data**
- **transform** is called on both training and test data using the training-fit parameters

**Example**: `scaler.fit(X_train)` computes mean and std from training. `scaler.transform(X_test)` subtracts the training mean and divides by training std. Test data never influences the scaling parameters.

## Order of Operations

Within a pipeline, transformations must happen in the correct order. A typical order:

```
1. Handle missing values (impute or drop)
2. Encode categorical variables
3. Create new features (ratios, interactions, aggregations)
4. Transform distributions (log, Box-Cox)
5. Scale/normalize numerical features
6. (Optional) Feature selection
```

**Why this order**: Imputation first ensures no missing values cause errors in later steps. Encoding before scaling ensures all features are numerical before scaling. Feature creation before scaling ensures new features are also scaled.

## Reproducibility

A pipeline ensures that running the same code twice produces the same results. This matters for:

- **Auditing**: A regulator asks "how did you compute this feature?" — the pipeline code is the answer
- **Debugging**: A model fails in production — you can replay the pipeline on production data
- **Collaboration**: A teammate needs to replicate your results — they run the pipeline
- **Deployment**: The same pipeline used in training must be deployed with the model

**Example**: In production, a new transaction arrives via API. The pipeline transforms it exactly as training data was transformed — same imputation values, same encoder mapping, same scaler parameters. Without a pipeline, you'd manually re-apply each step and risk inconsistency.

## Avoiding Leakage via Pipelines

A well-designed pipeline is the best defense against data leakage.

**How pipelines prevent leakage:**
- All statistics (mean, min, max, vocabularies) are computed **only on training data** during `fit`
- Test data is transformed using training statistics, never influencing them
- The pipeline object contains no information about test data
- Cross-validation works correctly: within each fold, the pipeline is re-fitted on the fold's training data

**Example**: Target encoding in a pipeline using scikit-learn's `TargetEncoder` — within cross-validation, each fold computes target means only on that fold's training data. Without a pipeline, you'd leak by computing target means on the full dataset before splitting.

## Real-World Implementation Pattern

```
Pipeline:
  1. SimpleImputer(strategy='median')       → fill missing age, income
  2. OneHotEncoder(handle_unknown='ignore')  → encode city, profession
  3. ColumnTransformer                       → apply different transforms to different columns
  4. StandardScaler                          → scale all numerical features
  5. LogisticRegression (or any model)       → train on fully processed data
```

The entire chain — from raw data to predictions — is one object. `pipeline.fit(X_train, y_train)` handles all steps. `pipeline.predict(X_test)` processes test data identically.

**Example**: A fraud detection pipeline: impute missing merchant category (mode), encode merchant ID (frequency encoding), create `tx_amount_ratio_to_user_avg` (interaction), log-transform amount, scale all features, then train XGBoost. All in one pipeline. Deploy the same pipeline to production.
