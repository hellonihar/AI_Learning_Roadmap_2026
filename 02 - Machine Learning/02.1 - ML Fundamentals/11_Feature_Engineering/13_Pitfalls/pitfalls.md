# Feature Engineering Pitfalls (Must-Know)

## 1. Data Leakage Through Features

The most common and most dangerous pitfall. Features that accidentally encode information from the future or from the target.

**Examples:**
1. **Target encoding on full dataset**: Computing mean default rate by city using ALL data (including test set). The test set's defaults influence training city encodings. Training accuracy looks great. In production (where you don't know defaults ahead), the model fails.
2. **Aggregation including current row**: Creating `avg_transaction_amount_per_user` but including the transaction itself in the average. At prediction time, you don't know the current transaction's amount yet.
3. **Using the target to create features**: In a loan default model, creating a feature "number_of_late_payments" — but late payments occur after the loan is given, overlapping with the default window.

## 2. Encoding Using Full Dataset

Fitting a label encoder, one-hot encoder, or frequency encoder on the FULL dataset before splitting. The test set influences the encoding vocabulary/statistics.

**Why it's dangerous**: Test data categories leak into training encoding. A category that appears only in the test set might get a specific encoding that affects training. In production, unseen categories break the encoder.

**Fix**: Fit encoding only on training data. Use `handle_unknown='ignore'` in OneHotEncoder to handle test-only categories gracefully.

## 3. Scaling Using Test Data

Computing mean, std, min, or max for scaling **before splitting** or **including test data in the computation**.

**Why it's dangerous**: Test data distribution influences training scaling. If test has extreme outliers, training scaler adjusts to accommodate them — indirectly leaking test information into the model.

**Fix**: `scaler.fit(X_train)` only. Transform X_test using training parameters.

## 4. Over-Engineering Features

Creating hundreds of features because "more is better." This leads to:
- Overfitting (model finds spurious correlations in noise features)
- Slow training and inference
- Harder to debug, deploy, and maintain
- Feature explosion: N features → N² interactions → N³ polynomials → unmanageable

**Example**: Creating price_sqft, price_sqft_log, price_sqft_sqrt, price_to_bedrooms, price_to_bathrooms, price_to_age, price_to_school_rating,... 50 derived features from 5 base features. The model overfits to noise in the derived features. A simpler feature set of 10 well-chosen features would perform better.

**Fix**: Start with domain-driven features (10-20). Add more only if they demonstrably improve validation performance. Regularly evaluate feature importance and remove unused features.

## 5. Feature Explosion

One-hot encoding high-cardinality categories or creating all possible interactions leads to an unmanageable number of features.

**Examples:**
1. **One-hot ZIP code (30K categories)**: 30K new features, most sparse (only 1 non-zero per row). Models train slowly, overfit to rare ZIP codes, and fail on unseen ZIPs.
2. **All pairwise interactions (100 features → 4,950 pairs)**: Most interactions are noise. The model has 5,000 features to fit with maybe 10K samples — guaranteed overfitting.
3. **Polynomial features (degree 3 on 20 features)**: Creates 1,770 features. Many are highly correlated, most are useless.

**Fix**: Use frequency/target encoding for high cardinality. Use domain knowledge to choose specific interactions rather than generating all possible. Regularization (L1) can help, but prevention is better.

## 6. Ignoring Domain Knowledge

The ML model doesn't know what the data means. It doesn't know that debt-to-income ratio is a standard creditworthiness metric or that price/sqft matters in real estate.

**Examples:**
1. **Raw transaction amounts without user context**: A \$500 transaction is normal for one user, suspicious for another (who usually spends \$20). Without normalizing by user average (domain insight), the model misses this.
2. **Raw timestamp without cyclic components**: A model given raw timestamp doesn't know that seasonal patterns exist. December (12) and January (1) appear far apart numerically but are adjacent in reality.
3. **Missing domain-specific ratios**: A real estate model given sqft and bedrooms separately but not price_per_sqft or rooms_per_sqft — the model must rediscover these relationships. Why make it learn what domain experts already know?

**Fix**: Before writing code, ask domain experts: "What signals do you use to make this decision?" Then engineer features that capture those signals.

## Quick Reference

```
❌ Fit scaler/encoder on full data before split
❌ Create features using the target or future information  
❌ One-hot encode 10K+ categories
❌ Generate all possible interactions/polynomials blindly
❌ Skip domain expert input
❌ Keep features that never get nonzero coefficients

✅ Fit statistics only on training data
✅ Create features that would be available at prediction time
✅ Use frequency/target encoding for high cardinality
✅ Start simple, add features only if validated
✅ Ask domain experts "what matters?"
✅ Regularize or remove features with zero importance
```
