# Dataset — Customer Churn Prediction

## Source
- **Telcom Customer Churn** (IBM Sample) — 7,043 customers, 21 columns.
- Supplemented with synthetic subscription data for scale (100k customers, 50 features).

## Features

### Demographics & Account
- `gender`, `senior_citizen`, `partner`, `dependents`
- `tenure_months`, `contract_type` (month-to-month / 1yr / 2yr)
- `payment_method`, `monthly_charges`, `total_charges`

### Service Features
- `phone_service`, `multiple_lines`, `internet_service`
- `online_security`, `online_backup`, `device_protection`, `tech_support`
- `streaming_tv`, `streaming_movies`

### Engineered Features (critical)
- `avg_monthly_charge` = total_charges / tenure
- `service_count` = sum of active services
- `tenure_segment` = bin(tenure, [0,6,12,24,48])
- `charge_trend` = slope of monthly charges over last 3 months
- `support_tickets_30d`, `support_tickets_90d`

## Target
- `churn`: Yes (1) / No (0).

## Class Balance
- No churn: ~73%
- Churn: ~27%
- Moderate imbalance; stratification sufficient.

## Known Challenges
- `TotalCharges` has 11 missing values — impute with `monthly_charges * tenure`.
- Month-to-month contracts dominate churn — model may over-learn this.
- Demographic-only model has low predictive power — feature engineering is where the value comes from.
