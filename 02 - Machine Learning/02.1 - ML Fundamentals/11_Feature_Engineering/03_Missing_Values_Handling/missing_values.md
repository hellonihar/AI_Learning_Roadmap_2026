# Missing Values Handling

## Why Missing Values Occur

- **Data not collected**: A customer didn't fill in their income on a form
- **Data not applicable**: "Number of pregnancies" is missing for male patients
- **Data lost**: Sensor failed, database migration dropped records
- **Data not yet available**: "Days to delivery" missing for orders still in transit

### Missing Completely at Random (MCAR) vs Not Random

- **MCAR**: Missingness is random — a sensor randomly failed 2% of the time. Safe to drop or impute without bias.
- **Missing Not at Random (MNAR)**: Missingness depends on the value itself — people with very high income are less likely to report it. Removing or imputing these creates bias.

**Example**: In a loan dataset, customers with missing income data are more likely to default (they didn't report because they were embarrassed). Simply imputing the mean income would hide this signal. The missingness itself is informative.

## Deletion Strategies

### Row Deletion (Listwise)
Remove any row with a missing value.

- **When to use**: Missingness is random and involves <5% of rows
- **Risk**: Losing valuable data, introducing bias if missingness is not random

### Column Deletion
Remove any column with too many missing values.

- **When to use**: Feature is missing >60-70% of values
- **Trade-off**: Lose potentially useful signal

**Example**: A feature "credit_score" is missing for 80% of loan applicants. Dropping the column is safer than imputing 80% of values with guesses.

## Simple Imputation

### Mean / Median / Mode
Replace missing value with the column's central tendency.

- **Mean**: Use for normally distributed numerical features
- **Median**: Use for skewed numerical features (income, transaction amount)
- **Mode**: Use for categorical features

**Example**: House price dataset missing 5% of `sqft` values. Impute with median sqft (1,500). Simple and works for low missingness.

### Constant Value Imputation
Replace with a fixed value (0, -1, "Unknown").

- **When to use**: Missingness has meaning — `0` for "no transactions yet", `-1` for "not applicable"
- **Risk**: Can distort distributions if overused

**Example**: In a transaction dataset, `days_since_last_purchase` is missing for new customers. Impute as -1 (or a high value like 999) to distinguish "new" from "inactive".

## Flagging Missingness as a Feature

Create an indicator column: `is_missing_income = 1` if income was missing, 0 otherwise.

**Why**: The pattern of missingness can be predictive. Customers who don't report income may be higher risk. Sensors that fail may indicate environmental conditions.

**Example**: Medical dataset: `blood_pressure` is missing for 15% of patients. Creating `is_missing_bp = 1` captures that patients with missing BP tend to be emergency cases with different outcomes.

## When NOT to Impute

| Situation | Action |
|---|---|
| Missingness is informative | Flag + leave as gap (handle in model) |
| Tree-based models | Can handle missing values natively (e.g., XGBoost learns split direction for missing) |
| Missing rate > 50% | Drop the column |
| Test set missing a feature entirely | Can't impute — need pipeline fallback |

**Example**: A gradient boosting model for fraud detection. Missing "card_verification_code" is highly predictive of fraud (fraudsters often skip this field). Imputing the mode would destroy this signal. Better to leave as-is and let the tree branch on missingness.
