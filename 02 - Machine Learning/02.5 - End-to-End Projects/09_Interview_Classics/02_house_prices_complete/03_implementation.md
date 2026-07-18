# Implementation — House Prices

## Pipeline

1. Feature engineering: missing → 0/"None", skew correction, interactions
2. Elastic Net (handles multicollinearity among 79 features)
3. Gradient Boosting (captures non-linear interactions)
4. Stacking: Ridge meta-model on out-of-fold predictions

## Code

### 1. Feature engineering

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import skew

df = pd.read_csv("train.csv")
y = np.log1p(df["SalePrice"])
X = df.drop(["Id", "SalePrice"], axis=1)

# --- Missing: categorical -> "None", numeric -> 0 ---
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].fillna("None")
    else:
        X[col] = X[col].fillna(0)

# --- Skew correction ---
numeric_cols = X.select_dtypes(include=[np.number]).columns
skewed = X[numeric_cols].apply(lambda x: skew(x.dropna()))
X[numeric_cols] = X[numeric_cols].apply(
    lambda x: np.log1p(x) if skew(x.dropna()) > 0.75 else x
)

# --- Interactions ---
X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
X["TotalBath"] = X["FullBath"] + 0.5 * X["HalfBath"] + X["BsmtFullBath"] + 0.5 * X["BsmtHalfBath"]
X["Age"] = X["YrSold"] - X["YearBuilt"]
X["RemodelAge"] = X["YrSold"] - X["YearRemodAdd"]
X["HasPool"] = (X["PoolArea"] > 0).astype(int)
X["HasGarage"] = (X["GarageArea"] > 0).astype(int)
X["HasBasement"] = (X["TotalBsmtSF"] > 0).astype(int)
X["TotalPorchSF"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
X["OverallGrade"] = X["OverallQual"] * X["OverallCond"]

# --- Encode ---
cat_cols = X.select_dtypes(include=["object"]).columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
```

### 2. Elastic Net + GBM

```python
enet = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=42)
gbm = GradientBoostingRegressor(
    n_estimators=500, max_depth=4, min_samples_leaf=5,
    learning_rate=0.01, subsample=0.8, random_state=42
)

# Cross-val scores
from sklearn.model_selection import cross_val_score
for name, model in [("ElasticNet", enet), ("GBM", gbm)]:
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
    print(f"{name}: CV RMSE = {-scores.mean():.4f}")
```

### 2. Stacking

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_enet = np.zeros(len(X))
train_gbm = np.zeros(len(X))

for tr_idx, va_idx in kf.split(X):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    enet.fit(X_tr, y_tr)
    train_enet[va_idx] = enet.predict(X_va)

    gbm.fit(X_tr, y_tr)
    train_gbm[va_idx] = gbm.predict(X_va)

# Meta model
meta_X = np.column_stack([train_enet, train_gbm])
meta = Ridge(alpha=1.0)
meta.fit(meta_X, y)

# Full train for test prediction
enet.fit(X, y)
gbm.fit(X, y)

# CV score
meta_preds = meta.predict(meta_X)
rmse = np.sqrt(mean_squared_error(y, meta_preds))
print(f"Stacking CV RMSE: {rmse:.4f}")
```
