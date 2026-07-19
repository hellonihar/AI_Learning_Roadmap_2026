# Time Series Basics for Deep Learning

## What Makes Time Series Different

A time series is a sequence of observations recorded over time. Unlike standard supervised learning where samples are independent, time series data carries a temporal dependency — what happened at time `t` is related to what happened at `t-1`, `t-2`, and so on. Deep learning models exploit these dependencies, but only if we prepare the data correctly.

## Core Components: Trend, Seasonality, and Noise

Every time series can be decomposed into three components:

- **Trend**: The long-term direction of the series (upward, downward, or flat). For example, monthly airline passenger counts show a clear upward trend over decades.
- **Seasonality**: Repeating patterns at fixed intervals — daily, weekly, yearly. Ice cream sales spike every summer; retail traffic surges every December.
- **Noise (Residual)**: The random variation that remains after removing trend and seasonality. Good models capture trend + seasonality and leave only noise as error.

Classical decomposition is additive (y = Trend + Seasonal + Noise) when the seasonal amplitude is constant, or multiplicative (y = Trend × Seasonal × Noise) when it grows with the trend.

## Stationarity

A stationary time series has constant mean, variance, and autocorrelation over time. Most DL models (and all statistical ones like ARIMA) assume or require stationarity.

**How to check:**
- **Visual inspection**: Does the mean drift? Does the variance change?
- **Augmented Dickey-Fuller (ADF) test**: Null hypothesis = series is non-stationary. A low p-value (< 0.05) rejects non-stationarity.

**How to make a series stationary:**
- **Differencing**: Subtract `y_t - y_{t-1}`. First-order differencing removes linear trend; second-order removes quadratic trends.
- **Log transformation**: Stabilizes variance for multiplicative patterns.
- **Seasonal differencing**: Subtract `y_t - y_{t-m}` where `m` is the seasonal period.

Deep learning models (LSTMs, CNNs) are more tolerant of non-stationarity than classical methods, but differencing still helps.

## Autocorrelation

Autocorrelation measures how a value correlates with its own past lags:

- **ACF (Autocorrelation Function)**: Plots correlation at lags 1, 2, 3, ... Helps identify MA terms in ARIMA and the seasonal period.
- **PACF (Partial Autocorrelation Function)**: Correlation after removing intermediate lags. Helps identify AR terms.

In deep learning, ACF/PACF plots guide the choice of **lookback window size** — the number of past steps the model should see.

## Lag Features

Lagged values of the target are the most powerful features for time series forecasting:

```python
df['lag_1'] = df['target'].shift(1)
df['lag_7'] = df['target'].shift(7)  # weekly lag
```

Additional derived features include:
- **Rolling statistics**: moving average, moving std, min/max over a window.
- **Date/time features**: hour of day, day of week, month, quarter, is_weekend.
- **Domain-specific**: holiday flags, promotional periods, weather data.

## Windowing (Sliding Window)

Time series data must be converted into supervised learning samples. Windowing creates pairs of (input window, target window):

```
Sequence: [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9]
Window size = 3, horizon = 1:
  X = [t0,t1,t2] → y = [t3]
  X = [t1,t2,t3] → y = [t4]
  X = [t2,t3,t4] → y = [t5]
  ...
```

For multi-step forecasting (horizon > 1):
```
  X = [t0,t1,t2] → y = [t3, t4, t5]
```

The lookback window (often called the **encoder length**) and the forecast horizon (**decoder length**) are key hyperparameters. The window size should capture at least one full seasonal cycle.

## Train/Test Splitting — No Leakage

Time series splits must respect temporal order. **Never use random shuffling** — it leaks future information into training.

```
|----- Train -----|---- Val ----|---- Test ----|
t0  ...  t700  t701  ...  t800  t801  ...  t900
```

The simplest approach is a chronological split (e.g., 80% train, 10% validation, 10% test). For smaller datasets, **walk-forward validation** (time series cross-validation) is more robust:

```
Fold 1: [train t0..t500] [val t501..t600]
Fold 2: [train t0..t600] [val t601..t700]
Fold 3: [train t0..t700] [val t701..t800]
```

Each fold expands the training set to include previous validation data. This mimics how models are used in production — always forecasting the future from the past.

**Critical warning**: When creating lag features or rolling statistics, compute them *within each fold* to avoid leakage from the validation set into training.

## Summary for the DL Practitioner

1. Decompose your series to understand its structure.
2. Check for stationarity; difference if needed (or let the model handle it).
3. Use ACF/PACF to estimate a sensible window size.
4. Engineer lag features and calendar features.
5. Create sliding windows with `torch.utils.data.Dataset`.
6. Split chronologically — never shuffle time series data.
7. Use walk-forward validation for robust evaluation.

These steps are the prerequisite for every deep learning architecture in this section.
