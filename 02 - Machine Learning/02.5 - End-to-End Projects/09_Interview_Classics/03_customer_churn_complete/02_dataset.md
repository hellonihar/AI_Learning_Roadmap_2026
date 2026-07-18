# Dataset — Customer Churn

## Source
Kaggle — Telco Customer Churn.  
[Link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Size
- 7,043 rows, 21 columns
- No missing values (except TotalCharges has 11 blank → coerce to numeric)

## Features

| Feature | Type | Description |
|---------|------|-------------|
| customerID | text | Unique ID |
| gender | binary | Male / Female |
| SeniorCitizen | binary | 0 or 1 |
| Partner | binary | Yes / No |
| Dependents | binary | Yes / No |
| tenure | numeric | Months of service |
| PhoneService | binary | Yes / No |
| MultipleLines | categorical | No phone service / No / Yes |
| InternetService | categorical | DSL / Fiber optic / No |
| OnlineSecurity | categorical | Yes / No / No internet |
| OnlineBackup | categorical | Yes / No / No internet |
| DeviceProtection | categorical | Yes / No / No internet |
| TechSupport | categorical | Yes / No / No internet |
| StreamingTV | categorical | Yes / No / No internet |
| StreamingMovies | categorical | Yes / No / No internet |
| Contract | categorical | Month-to-month / One year / Two year |
| PaperlessBilling | binary | Yes / No |
| PaymentMethod | categorical | Electronic check / Mailed check / etc. |
| MonthlyCharges | numeric | Monthly bill amount |
| TotalCharges | numeric | Total amount charged (11 missing) |

## Target
- `Churn`: Yes / No (27% churn rate)

## Known Challenges
- **Class imbalance** — 73% no churn, 27% churn.
- **11 missing TotalCharges** — these are new customers (tenure = 0), fill with 0.
- **High cardinality** in some categoricals (PaymentMethod has 4 levels).
- **Collinearity** — MonthlyCharges × tenure ≈ TotalCharges.
