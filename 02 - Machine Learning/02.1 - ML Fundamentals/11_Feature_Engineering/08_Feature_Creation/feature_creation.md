# Feature Creation (High-Impact Section)

Creating new features from raw data is where domain knowledge meets ML. The best features come from understanding what the data **means**, not just what it contains.

## Domain-Driven Feature Creation

Think about what **really matters** in your problem, then find ways to express it numerically.

**Examples:**
1. **E-commerce — "customer engagement"**: Raw data has login timestamps, page views, time on site. Domain insight: engagement = recency × frequency × duration. Engineer: `engagement_score = (days_since_last_login^-1) × log(avg_session_min) × log(pages_per_session)`. This single feature captures more signal than 10 raw behavioral columns.
2. **Lending — "creditworthiness"**: Raw data has income, existing loans, payment history. Domain insight: lenders care about repayment ability relative to existing obligations. Engineer: `debt_to_income_ratio = total_monthly_debt / monthly_income`, `credit_utilization = total_credit_used / total_credit_limit`. These ratios are standard in banking for a reason — they capture financial health better than raw numbers.
3. **Healthcare — "patient deterioration risk"**: Raw data has vital signs over time. Domain insight: trend matters more than absolute values. Engineer: `heart_rate_trend_slope_6h` (linear regression slope of HR over last 6 hours), `bp_variability_score` (standard deviation of blood pressure in 24h window). A stable patient at 90 BPM is fine; a patient whose HR went from 70 → 90 → 110 in 6 hours needs attention.

## Ratios and Differences

Combine two related features into one that captures their relationship.

**Examples:**
1. `price_per_sqft = price / square_footage` — captures value density. A \$500K house at 1,000 sqft (\$500/sqft) is very different from \$500K at 2,500 sqft (\$200/sqft).
2. `time_since_last_purchase_days` — recency is one of the strongest predictors in customer analytics
3. `support_tickets_this_month — support_tickets_last_month` — trend direction: worsening (positive) or improving (negative)?

## Aggregated Features (Group By)

Group data by an entity and compute statistics. These capture behavioral patterns.

**Per-user aggregations from transaction logs:**
- `avg_transaction_amount_30d`
- `max_transaction_amount`
- `transaction_count_7d` (velocity)
- `std_transaction_amount` (variability)
- `unique_merchant_categories_30d` (diversity)

**Examples:**
1. **Fraud detection**: `tx_count_last_24h` captures velocity — a card used 50 times in 24 hours is suspicious regardless of individual transaction amounts.
2. **Customer churn**: `login_count_last_7d / login_count_prev_7d` captures engagement change — dropping from 10 to 2 logins is a stronger churn signal than 2 logins alone.
3. **Recommendation**: `avg_rating_given_by_user`, `avg_rating_received_by_item`, `rating_count_by_user` — capture user strictness and item popularity.

## Interaction Features (Feature × Feature)

When the effect of one feature depends on another, create their product.

**Examples:**
1. `sqft × bedrooms` — a 2,000 sqft house with 2 bedrooms is spacious; with 6 bedrooms it's cramped. The interaction captures "space per room."
2. `income × has_loan` — high income with no loan is safe; high income with huge loan might be risky. The combination matters more than either alone.
3. `is_weekend × hour_of_day` — 3 AM on a weekday vs 3 AM on a weekend Saturday have very different meanings for fraud detection. The interaction captures this.

**Tree models handle interactions automatically** (a tree creates splits on different features in sequence). Linear models do not — you must add interactions explicitly.

## Binning / Discretization

Convert continuous features into discrete buckets.

**Why bin?**
- Captures non-linear relationships without changing the model
- Reduces noise from small fluctuations
- Makes features more interpretable

**Examples:**
1. **Age → age groups**: `0-12 (child)`, `13-19 (teen)`, `20-35 (young adult)`, `36-55 (mid)`, `56+ (senior)`. The model doesn't see 31 vs 32 as important — it sees "young adult" vs "mid."
2. **Income → income brackets**: `<25K`, `25-50K`, `50-100K`, `100-200K`, `200K+`. A linear model can't easily learn that the effect of income changes at 50K.
3. **Time of day → time buckets**: `morning (6-12)`, `afternoon (12-18)`, `evening (18-24)`, `night (0-6)`. Reduces 24 individual hours to 4 categories with clear behavioral meaning.

**Trade-off**: Loses granularity. The exact age 32 and 33 are now in the same bin. Use when the exact value matters less than the category.

## Polynomial Features

Create squared, cubed, or product terms: `x²`, `x³`, `x₁ × x₂`.

**Intuition**: Help linear models learn curves. A line can't fit a U-shape, but `y = w₁x + w₂x²` can (parabola).

**Example**: Ice cream sales vs temperature — sales increase up to a point, then decrease (too hot). A linear model would miss this. Adding `temperature²` allows the model to learn the peak. The input becomes: `sales = w₁ × temp + w₂ × temp² + b`. If w₁ > 0 and w₂ < 0, the parabola peaks.

## When to Use Each

| Technique | Best For | Risk |
|---|---|---|
| Ratios / differences | Normalizing by related variable | Dividing by zero |
| Aggregations | Behavioral data, logs | Data leakage (using future data in aggregation) |
| Interactions | Feature dependencies, linear models | Feature explosion (N features → N² pairs) |
| Binning | Non-linear relationships | Losing information, arbitrary boundaries |
| Polynomials | Simple curves | Extrapolation goes wild outside training range |
