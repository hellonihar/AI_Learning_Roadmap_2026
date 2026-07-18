# Understanding Data Before Engineering

Before creating features, you must understand the data you're working with. The right engineering depends on the data type.

## Feature Types

### Numerical
Numbers that represent quantities. Two sub-types:

- **Continuous**: Any value in a range (price: $0 to $10M, temperature: -20°C to 50°C, time: 0 to 86400 seconds)
- **Discrete**: Countable, finite values (number of bedrooms: 1, 2, 3, 4, 5; support tickets: 0, 1, 2, ...)

### Categorical
Labels that represent groups with **no inherent order**.

**Examples:**
1. **City**: Mumbai, Delhi, Bangalore — no ordering, just categories. One-hot encoding typically works.
2. **Product category**: Electronics, Clothing, Food — unordered groups. A model shouldn't assume "Electronics > Clothing" numerically.
3. **Color**: Red, Green, Blue — categories, not quantities.

### Ordinal
Categories with a **natural order** but unknown spacing between levels.

**Examples:**
1. **Education level**: High School < Bachelor's < Master's < PhD — ordered, but the gap between levels isn't uniform.
2. **Customer satisfaction**: Very Unsatisfied < Unsatisfied < Neutral < Satisfied < Very Satisfied — order matters, numerical difference is unclear.
3. **T-shirt size**: S < M < L < XL — order known, precise measurement difference unknown.

### Text (High-Level)
Free-form strings that need special processing. Models can't read text directly.

**Examples:**
1. Product descriptions, customer reviews, support tickets
2. News articles, social media posts
3. Medical notes, legal documents

### Date & Time
Timestamps that encode temporal patterns. Models see them as numbers but need cyclic and feature extraction.

**Examples:**
1. Transaction timestamp → extract hour, weekday, month, holiday flag
2. User signup date → compute account age
3. Last purchase date → compute recency

## Feature Distributions

Understanding how features are distributed helps choose transformations.

- **Normal (bell curve)**: Ideal for many models — log transformations usually not needed
- **Skewed right**: Most values small, few very large (income, transaction amounts) → often needs log transform
- **Bimodal**: Two peaks (hour of restaurant reservations: peaks at noon and 7 PM) → keep as-is or create time-of-day buckets
- **Uniform**: Evenly spread → no transformation needed
- **Multimodal**: Multiple peaks → may need binning or interaction features

## Cardinality

**Cardinality** = number of unique categories in a categorical feature.

| Cardinality | Example | Challenge |
|---|---|---|
| Low (2-10) | Gender, Yes/No, Day of week | Easy — one-hot |
| Medium (10-100) | State, Product category | Manageable — one-hot or target encoding |
| High (100-10K) | ZIP code, Product ID, User ID | Needs special handling (aggregation, embedding) |
| Extreme (10K+) | IP address, Session ID | Usually should be aggregated or removed |

## Target Leakage Awareness

Before engineering, check: **does this information exist at prediction time?**

**Example**: You're predicting whether a loan will default. Your raw data includes "customer called to complain" — but this happens after the loan is issued. In production, you won't have this at prediction time. Any feature derived from it will leak future information.
