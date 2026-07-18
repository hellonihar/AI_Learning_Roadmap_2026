# Implementation — Medical Cost Prediction

## Pipeline

1. EDA — log-transform target, visualize smoker × age interaction
2. Compare: Linear Regression (log-charges), Gamma GLM, Gradient Boosting
3. Ensemble: average predictions from log-linear + GBM
4. Evaluate on original (untransformed) scale

## Code

### 1. Log-transform and baseline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("insurance.csv")
y = df["charges"]
X = df.drop("charges", axis=1)

# Log transform target
y_log = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

cat_cols = ["sex", "smoker", "region"]
num_cols = ["age", "bmi", "children"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first"), cat_cols)
])

pipe_lr = Pipeline([
    ("prep", preprocessor),
    ("lr", LinearRegression())
])

pipe_lr.fit(X_train, y_train)
y_pred_log = pipe_lr.predict(X_test)
y_pred = np.expm1(y_pred_log)  # back to original scale
```

### 2. Gradient Boosting (no log-transform needed)

```python
from sklearn.ensemble import GradientBoostingRegressor

pipe_gbm = Pipeline([
    ("prep", preprocessor),
    ("gbm", GradientBoostingRegressor(
        n_estimators=300, max_depth=4, min_samples_leaf=10,
        learning_rate=0.05, random_state=42
    ))
])

pipe_gbm.fit(X_train, np.expm1(y_train))  # train on original scale
y_pred_gbm = pipe_gbm.predict(X_test)
```

### 3. Ensemble and evaluate

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ensemble: simple average
y_pred_ensemble = (np.expm1(y_pred_log) * 0.4 + y_pred_gbm * 0.6)

mae = mean_absolute_error(np.expm1(y_test), y_pred_ensemble)
rmse = np.sqrt(mean_squared_error(np.expm1(y_test), y_pred_ensemble))
print(f"MAE: ${mae:,.0f}  RMSE: ${rmse:,.0f}")

# Check error by smoker status
results = X_test.copy()
results["actual"] = np.expm1(y_test)
results["pred"] = y_pred_ensemble
results["error"] = results["actual"] - results["pred"]
print(results.groupby("smoker")["error"].describe())
```
