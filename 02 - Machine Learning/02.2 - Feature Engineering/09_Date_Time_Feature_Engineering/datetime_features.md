# Date & Time Feature Engineering

Timestamps contain rich patterns, but models see them as large integers. You must extract the meaningful cycles.

## Extracting Year, Month, Day, Weekday

The most basic time features: break a timestamp into human-interpretable components.

| Component | Range | Why It Matters |
|---|---|---|
| Hour | 0-23 | Business hours vs night, rush hour patterns |
| Day of week | 0-6 | Weekend vs weekday behavior |
| Day of month | 1-31 | Payday effects (1st/15th), end-of-month patterns |
| Month | 1-12 | Seasonal effects (holidays, weather) |
| Quarter | 1-4 | Business reporting cycles, seasonal budgeting |
| Year | 2020-2026 | Long-term trends, drift detection |
| Is weekend | 0/1 | Binary work vs leisure distinction |

**Examples:**
1. **Retail sales prediction**: Sales peak on weekends, dip on Tuesdays, spike on 1st of month (payday), surge in December. A model given only the raw timestamp can't learn these patterns. Each component extracts a different signal.
2. **Fraud detection**: Fraud rates spike at 2-4 AM (when cardholders are asleep). `hour_of_day` captures this. But the exact hour matters less than "unusual hour for this customer" — combine with user's typical hours.
3. **Traffic prediction**: Rush hour (8-9 AM, 5-6 PM) is very different from midday. `is_rush_hour`, `is_holiday`, `day_of_week` are all needed.

## Cyclical Features (Sin/Cos Intuition)

Hours and months are **cyclic** — hour 23 is closer to hour 0 than hour 14. Raw numbers don't capture this.

**Problem**: As raw numbers: `|23 — 0| = 23` but the actual difference should be 1 hour. A model would think 23 is far from 0.

**Solution**: Map time components to a circle using sin and cos.

```
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

Now every hour maps to a point on a unit circle. 23 and 0 are close in (sin, cos) space.

**Examples:**
1. **24-hour pattern**: Midnight (23:59) and 00:01 should be treated as close. Raw hour: 23 vs 0 → far. Sin/cos: both map to nearly identical (sin, cos) values → close.
2. **12-month cycle**: December and January are adjacent months. Raw: month=12 vs month=1 → far. Sin/cos: nearly identical. The sin/cos encoding of month also captures mid-year (June/July) being far from year-end.
3. **Transaction patterns**: A recurring monthly bill on the 1st and a rent payment on the 1st are different. `day_sin + day_cos` captures "beginning of month" proximity, while raw `day_of_month` treats 31 far from 1.

## Time Since / Time Until Features

Compute duration between a timestamp and a reference point.

- `days_since_last_purchase` — recency
- `hours_since_last_login` — engagement
- `days_until_next_bill` — urgency
- `account_age_days` — tenure

**Examples:**
1. **Customer churn**: `days_since_last_login` is one of the strongest predictors. A user who hasn't logged in for 90 days is very different from one who logged in yesterday. Raw data might have login timestamps, but the model needs the engineered "recency" feature.
2. **Medical readmission**: `days_since_discharge` helps predict readmission risk. Risk is highest in the first 7 days, then decreases. The model can learn a decay function from this engineered feature.
3. **Subscription retention**: `days_until_billing_date` — a user who just paid is unlikely to cancel; a user whose billing date is tomorrow might be about to cancel if they haven't used the service.
4. **Equipment maintenance**: `hours_since_last_maintenance` predicts failure probability. A machine running 1000 hours since service is riskier than one serviced yesterday.

## Lag Features (Conceptual)

Use past values of a feature as a new feature. "What happened before predicts what happens now."

- `sales_lag_1day` = sales from 1 day ago
- `price_lag_7days` = price from 7 days ago
- `tx_count_lag_30d` = transaction count in the 30 days before this row's date

**Example**: Predicting tomorrow's stock price: yesterday's price, last week's price, and the price change over the last 7 days are all lag features. They encode the recent trend.

## Rolling Window Features

Aggregate over a sliding window of recent data.

- `sales_rolling_mean_7d` — average sales over last 7 days
- `tx_count_rolling_24h` — number of transactions in last 24 hours
- `price_rolling_std_30d` — price volatility over last 30 days

**Example**: Fraud detection — `tx_count_last_24h` is a rolling window feature. If your card is normally used 2 times per day and suddenly 50 transactions appear in 1 hour, the rolling count captures this velocity anomaly.

## Avoiding Time-Based Leakage

Time-based features are especially prone to leakage. **Never use future information to create current features.**

**Critical rules:**
1. When computing rolling features, the window must look **backward only**. A "7-day rolling average" for January 15 should use January 8-14 data, not January 15-21.
2. Lag features must be computed **within the training window** only. Test set lags should never reference training set time periods.
3. Use **time-based split** (not random) when validating — otherwise your feature engineering will peek into the future.

**Example**: Creating "average transaction amount in the next 7 days" as a feature at time T — this uses the future (T+1 to T+7) to predict what happens at T. Perfect training performance (of course future transactions correlate with current ones), zero real-world usefulness.
