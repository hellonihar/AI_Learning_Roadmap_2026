# Handling Outliers

## What Is an Outlier?

A data point that differs significantly from others. But not all outliers are equal.

### Outliers vs Rare but Valid Values

| Type | Description | Example |
|---|---|---|
| **Genuine outlier (data error)** | Mistake in recording | Temperature sensor reads 200°C when actual is 25°C |
| **Rare valid value** | Unusual but real | A \$10M house in a \$300K neighborhood |
| **Novelty / anomaly** | Different process generating it | Fraudulent transaction among legitimate ones |

**Treat them differently!** Data errors should be removed or corrected. Rare valid values might be important signal. Anomalies might be exactly what you want to detect.

## Visual Detection

- **Boxplots**: Points beyond 1.5× IQR above Q3 or below Q1 are flagged as outliers
- **Scatter plots**: Visual clusters vs isolated points — immediate pattern recognition
- **Histograms**: Long tails on one side indicate potential outliers in that direction

**Example**: Plotting transaction amounts: 99% of transactions are under $500, but a few are $5,000+. The histogram shows a long right tail. These aren't errors — they're high-value purchases. Capping might lose signal.

## Statistical Detection (Intuition Only)

- **Z-score**: Values with |z| > 3 are potential outliers (for roughly normal distributions)
- **IQR method**: Points below Q1 — 1.5×IQR or above Q3 + 1.5×IQR
- **Isolation Forest**: Algorithm that isolates anomalies by random feature splits

## Capping / Flooring (Winsorization)

Replace extreme values with a threshold value instead of removing them.

- Cap at 95th percentile: any value above → set to 95th percentile value
- Floor at 5th percentile: any value below → set to 5th percentile value

**Example**: Income feature — 99% of values are under $500K. One person reported $50M (likely a billionaire). Cap at $500K. The model still learns from this person (they're at the max) without the single extreme value distorting scaling and gradients.

## Removal vs Transformation

| Approach | When | Risk |
|---|---|---|
| Remove | Clear data error, sensor glitch | Losing valid rare cases |
| Cap/Floor | Extreme but real values | Losing distinguishing power |
| Log transform | Skewed data with natural long tail | Interpretation becomes harder |
| Keep | Outliers are the signal (fraud, anomalies) | May skew model for normal cases |

**Example**: Credit card fraud — outliers (very large transactions at unusual hours) are exactly the signal. Don't remove them. But if a \$0 transaction appears (data error for \$X.XX), remove it.

## When Outliers Help the Model

Outliers aren't always bad. Sometimes they're the most informative data points.

1. **Fraud detection**: Fraudulent transactions are outliers by nature — unusual amount, location, time, merchant. Removing them would leave only normal transactions, and the model would have nothing to learn.
2. **Medical diagnosis**: Rare disease cases are outliers in lab results. They look unusual compared to healthy patients. These outliers are critical for training a diagnostic model.
3. **Novelty detection in manufacturing**: An outlier in vibration sensor data might indicate a bearing about to fail. These rare patterns are the early warning signal. Cap or remove them → lose the warning.

## Practical Guidelines

1. **Start without outlier handling** — see how the model performs first
2. **Investigate outliers** — plot them, understand whether they're errors or signal
3. **For errors**: Remove or correct
4. **For valid extremes**: Use robust scaling, or cap at reasonable thresholds
5. **When outliers are the target**: Don't handle them — design the model to detect them
