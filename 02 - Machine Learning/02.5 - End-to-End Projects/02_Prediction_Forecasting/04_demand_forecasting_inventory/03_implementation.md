# Implementation — Demand Forecasting & Inventory

## Approach: ARIMA Baseline → XGBoost → Ensemble

## Step 1 — ARIMA Baseline per Product

```python
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

def arima_forecast_4_weeks(series):
    """Fit ARIMA on single SKU-store demand series."""
    model = ARIMA(series, order=(1, 1, 1),
                  seasonal_order=(1, 1, 1, 52))
    fitted = model.fit()
    return fitted.forecast(steps=4)
```

## Step 2 — XGBoost with Lag Features

```python
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

def build_features(df, max_lag=4):
    df = df.sort_values("week").copy()
    for lag in range(1, max_lag + 1):
        df[f"demand_lag_{lag}"] = df["units_demanded"].shift(lag)
    df["demand_ma_4"] = df["units_demanded"].rolling(4).mean()
    df["demand_std_4"] = df["units_demanded"].rolling(4).std()
    df["is_promo"] = (df["promo_discount"] > 0).astype(int)
    return df.dropna()

df_feat = build_features(df)
train = df_feat[df_feat["week"] < "2025-01-01"]
test  = df_feat[df_feat["week"] >= "2025-01-01"]

features = [c for c in df_feat.columns if c not in
            ["units_demanded", "week", "store_id", "sku_id"]]
X_train, y_train = train[features], train["units_demanded"]
X_test, y_test = test[features], test["units_demanded"]
```

## Step 3 — Train & Blend with ARIMA Residuals

```python
xgb = XGBRegressor(n_estimators=300, max_depth=5,
                   learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

# Blend: average XGBoost + ARIMA
arima_preds = arima_forecast_4_weeks(train["units_demanded"])
# Repeat ARIMA to match test length
arima_full = np.tile(arima_preds, len(test) // 4 + 1)[:len(test)]
blend = 0.7 * xgb_preds + 0.3 * arima_full

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, blend)
print(f"Blend MAE: {mae:.2f}")
```

## Step 4 — Safety Stock Calculation

```python
# Safety stock = Z × σ_demand × sqrt(lead_time)
lead_time_weeks = 2  # supplier lead time
z_score = 2.05       # 98% service level
error_std = np.std(y_test - blend)
safety_stock = int(z_score * error_std * np.sqrt(lead_time_weeks))
print(f"Safety stock per SKU-store: {safety_stock} units")
```
