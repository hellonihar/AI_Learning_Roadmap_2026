# Implementation — Energy Consumption Forecasting

## Step 1 — Load, Resample, Engineer Features

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data/household_power.txt", sep=";",
                  parse_dates={"dt": ["Date", "Time"]},
                  na_values=["?"], low_memory=False)
df = df.set_index("dt").resample("h").mean().ffill(limit=2)

df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Rolling features (24h window)
cols = ["Global_active_power", "Global_intensity"]
for col in cols:
    df[f"{col}_lag_1h"] = df[col].shift(1)
    df[f"{col}_lag_24h"] = df[col].shift(24)
    df[f"{col}_ma_24h"] = df[col].rolling(24).mean()

target = "Global_active_power"
df = df.dropna()
```

## Step 2 — Train/Test Split (Time-Based)

```python
train = df[df.index < "2010-01-01"]
test  = df[df.index >= "2010-01-01"]

features = ["hour", "day_of_week", "month", "is_weekend",
            "Global_active_power_lag_1h",
            "Global_active_power_lag_24h",
            "Global_active_power_ma_24h",
            "Global_intensity_lag_1h",
            "temperature", "humidity"]

X_train, y_train = train[features], train[target]
X_test, y_test   = test[features], test[target]
```

## Step 3 — Train XGBoost

```python
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

model = XGBRegressor(n_estimators=500, max_depth=7,
                     learning_rate=0.05, subsample=0.8,
                     colsample_bytree=0.8, random_state=42)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.3f} kWh")
```

## Step 4 — Feature Importance & Hourly Breakdown

```python
# Per-hour accuracy
test["pred"] = y_pred
test["hour"] = test.index.hour
hourly_rmse = (test["Global_active_power"] - test["pred"]) \
    .groupby(test["hour"]).apply(lambda x: np.sqrt(np.mean(x**2)))
print(hourly_rmse)

# Plot top features
feat_imp = pd.Series(model.feature_importances_,
                     index=features).sort_values()
```
