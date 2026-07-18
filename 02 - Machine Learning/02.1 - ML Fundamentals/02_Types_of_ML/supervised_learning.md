# Supervised Learning

Model learns from **labeled** data: input → output mappings.

## How It Works

- Training data: `(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)`
- The model learns to predict `y` from `x`
- After training, it predicts `y` for new, unseen `x`

## Two Main Flavors

### Regression
Predict a **continuous** number.

**Examples:**
1. **House price prediction**: Given sq ft, bedrooms, location → predict sale price (e.g., $450K)
2. **Weather forecasting**: Given pressure, humidity, wind → predict tomorrow's temperature in °C
3. **Stock price forecasting**: Given historical prices, volume, news sentiment → predict next day's closing price

### Classification
Predict a **discrete category**.

**Examples:**
1. **Email spam detection**: Input is email text → output is "spam" or "not spam"
2. **Medical diagnosis**: Input is patient vitals + lab results → output is "diabetic" or "non-diabetic"
3. **Image classification**: Input is pixels → output is "cat", "dog", "bird", etc.

## Common Algorithms

| Algorithm | Type | When to Use |
|---|---|---|
| Linear/Logistic Regression | Regression / Binary Class | Baseline, interpretable |
| Decision Trees | Both | Non-linear, interpretable |
| Random Forest | Both | Robust, handles missing data |
| SVM | Both | High-dimensional, clear margin |
| Neural Networks | Both | Complex patterns, large data |
| k-NN | Both | Simple, non-parametric |
