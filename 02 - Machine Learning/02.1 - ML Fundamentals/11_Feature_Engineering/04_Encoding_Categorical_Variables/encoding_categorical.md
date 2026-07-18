# Encoding Categorical Variables

## Why Models Can't Handle Categories Directly

Models do math (matrix multiplication, weighted sums). Math needs numbers. "Mumbai" × weight + "Delhi" × weight doesn't compute. Every category must be converted to numbers.

## Label Encoding

Assign each category a unique integer: Mumbai=0, Delhi=1, Bangalore=2, Chennai=3.

**When it works**: Ordinal categories where order matters. `T-shirt size: S=0, M=1, L=2, XL=3`.

**When it breaks**: Nominal (unordered) categories. The model interprets Bangalore > Mumbai because 2 > 0. This implies an order relationship that doesn't exist.

**Example**: Encoding `city` as 0-50. The model might learn "higher city number = higher fraud rate" just because Mumbai got 0 and Delhi got 1 — nonsense. Linear models especially suffer from this. Tree models are less affected (they can split at arbitrary thresholds).

## One-Hot Encoding

Create a binary column for each category. Each row has exactly one "1" and the rest "0".

```
Color     →  is_Red  is_Blue  is_Green
Red       →    1        0        0
Blue      →    0        1        0
Green     →    0        0        1
```

**When to use**: Nominal categories with low cardinality (<10 categories).

**Trade-off**: Creates many columns. 50 cities = 50 new columns. 10K product IDs = 10K columns (prohibitive).

## Dummy Variable Trap

When one-hot encoding, if you keep all k columns, they are perfectly collinear (sum = 1 for every row). Linear models fail because they can't distinguish the intercept from the sum of all category coefficients.

**Fix**: Drop one category (k-1 columns). The dropped category becomes the "reference" level.

**Example**: Encoding `Red, Blue, Green` → create `is_Blue`, `is_Green`. If both are 0, it must be Red. Avoids collinearity.

## Ordinal Encoding

Assign integers that reflect the natural order.

```
Education:  High School=1, Bachelor's=2, Master's=3, PhD=4
Satisfaction: Very Unsatisfied=0, Unsatisfied=1, Neutral=2, Satisfied=3, Very Satisfied=4
```

**Using LabelEncoder from sklearn**: Maps categories to 0, 1, 2, ... n-1. Preserves order only if the category strings sort alphabetically (usually not). Better to manually define the mapping.

**Example**: `education_map = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}`. Ensures correct order regardless of alphabetical sorting.

## High-Cardinality Categorical Features (Conceptual)

When a category has hundreds or thousands of unique values, one-hot encoding is impractical.

**Approaches:**
1. **Frequency encoding**: Replace category with its count/frequency in the dataset. "Mumbai appears 5,000 times → replace 'Mumbai' with 5000."
2. **Target encoding**: Replace category with mean of the target for that category. "Average default rate for Mumbai customers is 3.2% → replace 'Mumbai' with 0.032." (⚠️ Risk of target leakage — must use cross-validation to compute safely.)
3. **Binary encoding**: Convert integers to binary strings, then one-hot the bits. Fewer columns than full one-hot.

**Examples:**
1. **ZIP code (30K unique values)**: One-hot would create 30K columns. Frequency encoding: `zip_code_popularity` = count of transactions from that ZIP. Captures "common vs rare locations" without exploding dimensions.
2. **User ID (1M users)**: One-hot is impossible. Frequency encoding: `user_activity_count`. Better: aggregate user behavior into numerical features (avg_spend, recency, etc.) and drop raw user ID.
3. **IP address in fraud detection**: One-hot is nonsense (each IP is unique). Frequency encoding: `tx_count_from_this_ip_last_24h`. Captures velocity — far more predictive than the raw IP string.

## Encoding Leakage Risks

**Critical**: Any encoding that uses target information (target encoding) or global statistics (frequency encoding on full data) must be done **inside cross-validation folds** or **after train-test split**.

**Example**: You compute `mean_default_rate_by_city` on the entire dataset (train + test). The test set's default rates influence the encoding of training cities. Model overfits to test data → inflated performance.

**Fix**: Fit encoding statistics **only on training data**. Apply the same mapping to test data. Never let test data influence encoding.
