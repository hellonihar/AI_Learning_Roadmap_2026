# Regression vs Classification

## Regression

Predict a **continuous numeric value**. The output is a real number.

**Examples:**
1. **Predicting house price**: Sq ft, bedrooms, location → **$485,000** (any value in a continuous range)
2. **Temperature forecasting**: Pressure, humidity, wind → **23.7°C** (not just "hot" or "cold")
3. **Estimating delivery time**: Distance, traffic, historical speed → **14.3 minutes**

## Classification

Predict a **discrete category**. The output is a class label.

**Examples:**
1. **Email classification**: Email text → **"spam"** or **"not spam"** (2 classes, binary)
2. **Handwritten digit recognition**: Pixel image → **"3"**, **"7"**, or **"9"** (10 classes)
3. **Disease diagnosis**: Patient vitals + lab results → **"no disease"**, **"early stage"**, **"advanced"** (3 ordered classes)

## Quick Rule of Thumb

| If your output is... | It's... |
|---|---|
| A number (integer or float) | Regression |
| A category | Classification |
| A probability | Classification (probabilistic output) |
| A rank/order | Can be either (ordinal regression or ranking) |

## When the Line Blurs

- **Ordinal regression**: Predicting "movie rating 1–5 stars" — ordered categories, could be treated as regression or multiclass classification
- **Probabilistic classification**: Model outputs "90% chance of rain" — a continuous number, but it's still a probability over classes
