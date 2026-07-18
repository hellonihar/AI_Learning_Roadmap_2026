# Customer Lifetime Value — Implementation

## Steps

1. **Recency/Frequency/T** matrix — Per customer: frequency (repeat purchases), recency (time of last purchase), T (time since first purchase), monetary_value
2. **BG/NBD Model** — Fit using `lifetimes` library to predict future purchase frequency
3. **Gamma-Gamma Model** — Fit on monetary value conditional on frequency (only customers with >1 purchase)
4. **CLV Prediction** — Expected purchases in next 6 months × expected average spend
5. **Baseline Regression** — LinearRegression, RandomForest on RFM features for comparison

## Key Code

```python
# Prepare RFM matrix for lifetimes
from lifetimes.utils import summary_data_from_transaction_data

rfm = summary_data_from_transaction_data(
    transactions, 'customer_id', 'transaction_date',
    monetary_value_col='amount', freq='D'
)
rfm = rfm[rfm['frequency'] > 0].copy()  # BG/NBD needs repeat purchases

print(rfm.describe())
# frequency, recency, T, monetary_value columns
```

```python
# BG/NBD model
from lifetimes import BetaGeoFitter

bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(
    rfm['frequency'], rfm['recency'], rfm['T'],
    verbose=True
)
print(bgf.summary)

# Expected purchases in next 180 days
t = 180
rfm['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    t, rfm['frequency'], rfm['recency'], rfm['T']
)

# Gamma-Gamma model for monetary value
from lifetimes import GammaGammaFitter

return_customers = rfm[rfm['frequency'] > 0]
ggf = GammaGammaFitter(penalizer_coef=0.0)
ggf.fit(
    return_customers['frequency'],
    return_customers['monetary_value'],
    verbose=True
)
rfm['predicted_avg_order'] = ggf.conditional_expected_average_profit(
    rfm['frequency'], rfm['monetary_value']
)
```

```python
# CLV = expected purchases × expected value
rfm['clv_6mo'] = rfm['predicted_purchases'] * rfm['predicted_avg_order']

# Actual evaluation: spend in holdout period (last 6 months)
holdout_spend = transactions[
    transactions['transaction_date'] >= '1998-01-01'
].groupby('customer_id')['amount'].sum()

from sklearn.metrics import mean_absolute_error
y_true = holdout_spend.reindex(rfm.index).fillna(0)
y_pred = rfm['clv_6mo']
mae = mean_absolute_error(y_true, y_pred)
print(f'BG/NBD + Gamma-Gamma MAE: ${mae:.2f}')
```
