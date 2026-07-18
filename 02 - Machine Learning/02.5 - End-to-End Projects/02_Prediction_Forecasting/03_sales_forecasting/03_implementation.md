# Implementation — Sales Forecasting

## Step 1 — Resample to Weekly & Add Calendar Features

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data/sales_daily.csv", parse_dates=["date"])
df = df.set_index("date").resample("W")["sales"].sum().reset_index()

# Calendar features
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["is_holiday_week"] = df["date"].isin(holiday_dates).astype(int)
df["snap_week"] = (df["date"].dt.day <= 7).astype(int)  # SNAP start

# Lag features (4 weeks)
for lag in [1, 2, 4, 8]:
    df[f"sales_lag_{lag}"] = df["sales"].shift(lag)

# Rolling means
df["sales_ma_4"] = df["sales"].rolling(4).mean()
```

## Step 2 — Time-Series Train/Test Split

```python
from sklearn.model_selection import TimeSeriesSplit

df = df.dropna()
train = df[df["date"] < "2016-01-01"]
test  = df[df["date"] >= "2016-01-01"]

feature_cols = ["week_of_year", "month", "is_holiday_week",
                "snap_week", "sales_lag_1", "sales_lag_2",
                "sales_lag_4", "sales_ma_4"]
X_train, y_train = train[feature_cols], train["sales"]
X_test, y_test   = test[feature_cols], test["sales"]
```

## Step 3 — XGBoost with Time-Series CV

```python
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

tscv = TimeSeriesSplit(n_splits=4)
mae_scores = []

for train_idx, val_idx in tscv.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = XGBRegressor(n_estimators=200, learning_rate=0.1,
                         max_depth=5, random_state=42)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              verbose=False)
    preds = model.predict(X_val)
    mae_scores.append(mean_absolute_error(y_val, preds))

print(f"CV MAE: {np.mean(mae_scores):.1f} ± {np.std(mae_scores):.1f}")
```

## Step 4 — Evaluate

```python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
wmape = np.sum(np.abs(y_test - y_pred)) / np.sum(y_test) * 100
print(f"Test MAE: {mae:.1f}, WMAPE: {wmape:.1f}%")
```
