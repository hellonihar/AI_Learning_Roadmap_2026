# CatBoost

**Categorical Boosting** — a gradient boosting library designed to handle **categorical features natively**, without manual encoding.

## What Makes CatBoost Special

### Ordered Boosting
Standard gradient boosting suffers from **target leakage** — each tree uses the same data that computed the residuals. CatBoost uses ordered boosting (similar to online learning) to compute unbiased residuals.

### Native Categorical Feature Handling
No more one-hot encoding. CatBoost applies an efficient **ordered target encoding** with permutation-based statistics, avoiding leakage while handling high-cardinality categories.

### Symmetric Trees
CatBoost builds balanced trees (both children at the same depth). This makes prediction faster and reduces overfitting.

## Key Hyperparameters

| Parameter | Effect |
|---|---|
| `iterations` | Number of trees |
| `learning_rate` | Shrinkage rate |
| `depth` | Tree depth (default 6) |
| `l2_leaf_reg` | L2 regularization |
| `border_count` | Discretization bins for numerical features |
| `cat_features` | Specify which features are categorical |

## When to Use CatBoost Over XGBoost/LightGBM

- **Lots of categorical features**: CatBoost handles them natively, outperforming manually encoded alternatives
- **Small datasets**: Ordered boosting reduces overfitting on small data
- **Default parameters work well**: CatBoost is the most "out-of-the-box" friendly boosting library
- **Need high accuracy with minimal tuning**

## Examples

1. **Insurance risk modeling**: Policy data is full of categorical features (car model, location, occupation, marital status). CatBoost handles 50+ categorical columns without any encoding, matching the performance of carefully tuned XGBoost with target encoding.
2. **E-commerce product ranking**: Features include category, brand, seller, color — all categorical with high cardinality. CatBoost natively handles categories with 10K+ unique values.
3. **Customer survey analysis**: Columns: region (100 values), industry (50 values), company_size (S/M/L/XL), satisfaction (1-5). CatBoost handles the mixed categorical/numerical data with minimal preprocessing.
