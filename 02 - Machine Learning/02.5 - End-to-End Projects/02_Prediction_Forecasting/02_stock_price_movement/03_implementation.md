# Implementation — Stock Price Movement

## Step 1 — Fetch & Engineer Features

```python
import yfinance as yf
import pandas as pd
import numpy as np

def engineer_features(ticker):
    df = yf.download(ticker, start="2000-01-01", end="2026-06-01")
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["Price_SMA_ratio"] = df["Close"] / df["SMA_10"]

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26

    # Target: tomorrow's direction
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df.dropna()
```

## Step 2 — Time-Based Train/Test Split

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, classification_report

df = engineer_features("AAPL")
cutoff = int(len(df) * 0.80)
train, test = df.iloc[:cutoff], df.iloc[cutoff:]

feature_cols = ["SMA_10", "SMA_50", "Price_SMA_ratio",
                "RSI_14", "MACD", "Volume"]
X_train, y_train = train[feature_cols], train["Target"]
X_test, y_test = test[feature_cols], test["Target"]
```

## Step 3 — Train with Class Weighting

```python
model = RandomForestClassifier(
    n_estimators=300, max_depth=6,
    class_weight="balanced", random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Precision (up): {precision_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))
```

## Step 4 — Feature Importance

```python
import matplotlib.pyplot as plt
imps = pd.Series(model.feature_importances_,
                 index=feature_cols).sort_values()
imps.plot.barh(title="Feature Importance")
plt.tight_layout()
```
