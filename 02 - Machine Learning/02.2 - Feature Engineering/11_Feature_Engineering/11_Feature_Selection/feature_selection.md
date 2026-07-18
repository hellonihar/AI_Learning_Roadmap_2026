# Feature Selection (Not Algorithms)

## Why Fewer Features Can Be Better

More features ≠ better model. Adding irrelevant or redundant features can hurt performance.

**Reasons to reduce features:**
1. **Curse of dimensionality**: As features increase, data becomes sparse — harder to find patterns
2. **Overfitting**: More noise features = more opportunities to fit spurious correlations
3. **Training time**: More features = slower training and inference
4. **Interpretability**: 10 features is explainable; 10,000 is not
5. **Deployment cost**: More features = more data pipelines, more failure points

## Redundant vs Informative Features

| Feature Type | Example | Effect |
|---|---|---|
| Informative | `credit_utilization` for loan default | Helps the model |
| Redundant | `credit_utilization` and `credit_utilization × 100` | No new info, adds noise |
| Irrelevant | `customer_shoe_size` for loan default | Pure noise — model might accidentally correlate |
| Leaky | `number_of_late_payments_during_default_period` | Destroys generalization |

**Example**: In house price prediction, `sqft` and `sqft_log` and `sqft_squared` and `sqft_cubed` are redundant (they all encode the same base signal). The model doesn't need all four — pick the best one or two.

## Correlation-Based Filtering

Remove features that are highly correlated with each other (redundant) while keeping those correlated with the target (informative).

**Rules of thumb:**
- If two features have |correlation| > 0.9, keep only one
- If a feature has near-zero correlation with the target, consider removing it (but be careful — it might still matter in interactions)

**Example**: In credit scoring, `number_of_open_loans` and `total_debt_outstanding` are highly correlated (more loans = more debt). Keep only `debt_to_income_ratio` which captures both and is more interpretable.

## Variance Thresholding (Intuition)

If a feature has very low variance (almost all values are the same), it carries almost no information.

- `is_alive` = 1 for 100% of rows → zero variance → remove
- `country_code` = "USA" for 99.9% of rows → near-zero variance → probably remove
- `zillow_price_change` = 1 for 25%, -1 for 25%, 0 for 50% → informative, keep

**Example**: In a global e-commerce dataset, 98% of transactions are in USD. The `currency` feature has near-zero variance (almost always "USD"). Removing it loses negligible information and reduces noise.

## Domain-Driven Selection

Sometimes you don't need data — you need expertise. Domain experts can identify which features are likely predictive and which are obvious noise.

**Examples:**
1. **Credit scoring**: A banker knows `debt_to_income_ratio`, `payment_history`, `credit_utilization` are the core predictors. `customer_favorite_color`, `pet_name`, `shoe_size` are noise — remove without looking at correlation.
2. **Healthcare**: A doctor knows `age`, `blood_pressure`, `cholesterol`, `smoking_status` predict heart disease. `hair_color`, `favorite_food` (without context) are noise.
3. **E-commerce**: A marketer knows `recency`, `frequency`, `monetary_value` (RFM) predict customer value. `browser_language`, `signup_source` might matter but are secondary — filter by domain knowledge first.

## Selection vs Dimensionality Reduction (Conceptual)

| | Feature Selection | Dimensionality Reduction |
|---|---|---|
| **What** | Remove features | Transform features into fewer new ones |
| **Output** | Subset of original features | New synthetic features |
| **Interpretability** | High (original features kept) | Low (new features are combinations) |
| **Examples** | Keep sqft, bedrooms, location | PCA components (no clear meaning) |
| **When** | When interpretability matters | When you just need lower dimension |
| **Method** | Correlation, domain knowledge, variance | PCA, t-SNE, autoencoders |

**Example**: PCA on 100 purchase-history features → 10 components that explain 90% of variance. You can't tell "what" each component means, but the model works. If you need to explain to a client why a loan was denied, stick with feature selection.

## Practical Workflow

1. Start with all features
2. Remove known noise (domain knowledge)
3. Check correlations — drop highly redundant features
4. Remove near-zero variance features
5. Let the model train — if it overfits, remove more features
6. Use regularization (L1) which does automatic feature selection during training
