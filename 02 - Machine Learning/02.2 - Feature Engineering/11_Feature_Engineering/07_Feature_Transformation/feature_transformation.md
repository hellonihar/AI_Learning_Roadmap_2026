# Feature Transformation

## Why Transform Features?

- Make skewed distributions more symmetric (models prefer normal-like distributions)
- Stabilize variance (reduce heteroscedasticity)
- Linearize relationships (make non-linear patterns easier for linear models to learn)
- Interpretability: "doubling income increases spending by 10%" vs "a \$1 increase in income increases spending..."

## Log Transformation

`x_transformed = log(x)`

**Effect**: Stretches small values, compresses large values. Right-skewed data becomes more normal.

**When to use**: Positive-only data with long right tails (income, transaction amounts, population counts, response times)

**Examples:**
1. **Income prediction**: Raw income distribution is heavily right-skewed (most people earn \$30-80K, a few earn millions). Log-transformed income becomes approximately normal — linear regression works much better.
2. **House price prediction**: House prices ($100K — $10M) — a \$100K error on a \$200K house is huge (50%), but a \$100K error on a \$5M house is tiny (2%). Log transformation makes errors proportionate: predicting log(price) means you're predicting percentage error, not absolute error.
3. **Word counts in text**: Some documents have 50 words, others have 5,000. Log-transform to compress the range. A model predicting from raw word counts would be dominated by document length.

**⚠️ Can't log negative values.** Must shift: `log(x + 1)` for zero-inflated data.

## Power Transformations

Family of transformations that raise data to a power. Two common approaches:

- **Box-Cox**: `(x^λ — 1) / λ` for λ ≠ 0, `log(x)` for λ = 0. Requires positive data.
- **Yeo-Johnson**: Like Box-Cox but handles zero and negative values.

Both automatically find the optimal λ to make the distribution most normal.

**Example**: Sensor readings with unusual distribution. Instead of guessing whether to use log, sqrt, or inverse, Box-Cox finds the best transformation automatically.

## Handling Skewed Distributions

| Skew Direction | Visual | Transformation |
|---|---|---|
| Right-skewed (positive skew) | Long tail on right | Log, sqrt, Box-Cox |
| Left-skewed (negative skew) | Long tail on left | Square, cube, exponential |
| Bimodal | Two peaks | No transformation — consider binning or mixture model |

**Example**: `days_to_purchase` — most customers buy within 1-7 days, some take 100+ days. Right-skewed. Log-transform: `log(days + 1)` compresses the extreme values and makes the distribution more symmetric.

## Non-Linear Transformations

Create `x²`, `x³`, `sqrt(x)`, `1/x` to capture non-linear relationships.

**Example**: Distance to city center — house prices drop rapidly near the center and plateau further out. `1/distance` captures this decay better than raw distance: a linear model with `1/distance` can learn "price increases as 1/distance increases" (which matches the actual market).

## Stabilizing Variance

Some features have variance that changes with the mean (heteroscedasticity). Transformations can help.

**Example**: Sales data — a popular product might have daily sales of 100±20 (variance 400), while a niche product might have 5±3 (variance 9). The variance scales with the mean. Log transformation: now both products have roughly constant variance on log scale.

## Interpretability Trade-Offs

| Transformation | Interpretability |
|---|---|
| `log(price)` | "1% increase in price" — harder for business stakeholders |
| `sqrt(area)` | Unnatural unit — "square root of square feet" |
| Box-Cox | Optimal λ might be 0.3 — completely uninterpretable |
| No transformation | "A \$10K price increase..." — clear |

**Rule**: Only transform if the model performance improves significantly. If the model is a simple linear regression for business reporting, avoid transformations that destroy interpretability. For deep learning or ensemble models, transformation matters less.
