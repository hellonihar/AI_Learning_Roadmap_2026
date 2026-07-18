# Manufacturing Fault Detection — Implementation

## Steps

1. **Sensor Selection** — Variance + correlation filter: remove sensors with >50% missing or near-zero variance; keep 148 sensors
2. **Missing Value Imputation** — Forward-fill + median imputation per sensor (sensors drift between readings)
3. **Rolling Window Features** — Compute rolling mean/std/min/max over last 5, 10, 25 wafers per sensor
4. **Dimensionality Reduction** — PCA to 30 components (explains 72% variance)
5. **Threshold Model** — Per-sensor moving average ± 3σ with adaptive window
6. **Ensemble** — Combine rule-based threshold violations + statistical deviation score

## Key Code

```python
# Rolling window feature engineering
def create_rolling_features(df, window_sizes=[5, 10, 25]):
    roll_features = pd.DataFrame(index=df.index)
    for w in window_sizes:
        for stat, func in [('mean', 'mean'), ('std', 'std'), ('min', 'min'), ('max', 'max')]:
            rolled = df[selected_sensors].rolling(window=w, min_periods=1)
            vals = getattr(rolled, func)().values
            # Add to feature set
            col_names = [f's{s}_{stat}_w{w}' for s in selected_sensors]
            roll_features = pd.concat([
                roll_features,
                pd.DataFrame(vals, columns=col_names, index=df.index)
            ], axis=1)
    return roll_features
```

```python
# Adaptive thresholding per sensor
thresholds = {}
for sensor in selected_sensors:
    vals = df[sensor].values

    # Rolling baseline (EWMA)
    baseline = pd.Series(vals).ewm(span=20, adjust=False).mean().values
    resid = vals - baseline

    # Adaptive sigma (rolling MAD for robustness)
    mad = pd.Series(np.abs(resid)).rolling(50, min_periods=1).median().values
    sigma = 1.4826 * mad  # MAD -> sigma

    upper = baseline + 3 * sigma
    lower = baseline - 3 * sigma
    thresholds[sensor] = {'upper': upper, 'lower': lower, 'baseline': baseline}

# Fault score: count of sensors exceeding adaptive thresholds
fault_scores = np.zeros(len(df))
for sensor, t in thresholds.items():
    fault_scores += ((df[sensor].values > t['upper']) | (df[sensor].values < t['lower'])).astype(int)
```

```python
# Time-based cross-validation
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = []
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(
        n_estimators=200, scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric='aucpr', use_label_encoder=False
    )
    model.fit(X_train, y_train)
    scores.append(model.predict_proba(X_val)[:, 1])
```
