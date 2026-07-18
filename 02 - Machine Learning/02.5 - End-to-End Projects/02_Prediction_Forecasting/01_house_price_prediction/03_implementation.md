# Implementation — House Price Prediction

## Pipeline Overview

```
Raw Data → Missing Impute → Encode → Feature Engine → Scale → Model → Log-Inverse
```

## Step 1 — Preprocessing & Log Target

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

df = pd.read_csv("data/train.csv")
y = np.log1p(df["SalePrice"])           # fix skew
X = df.drop(columns=["Id", "SalePrice"])

# Ordinal quality map
qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
for col in X.select_dtypes(include="object"):
    if X[col].iloc[0] in qual_map:
        X[col] = X[col].map(qual_map).fillna(0)
```

## Step 2 — Feature Engineering

```python
# Total square footage
X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
# Age at sale
X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
# Combining garage/basement finish
X["HasGarage"] = (X["GarageArea"] > 0).astype(int)
X["HasBsmt"] = (X["TotalBsmtSF"] > 0).astype(int)
# Polynomial on top-rated feature
X["OverallQual_SQ"] = X["OverallQual"] ** 2
```

## Step 3 — Modeling (Ridge → XGBoost → Ensemble)

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

num_cols = X.select_dtypes(exclude="object").columns
cat_cols = X.select_dtypes(include="object").columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

model = Pipeline([
    ("prep", preprocessor),
    ("xgb", XGBRegressor(n_estimators=500, learning_rate=0.05,
                         max_depth=5, random_state=42))
])

scores = cross_val_score(model, X, y, cv=5,
                         scoring="neg_root_mean_squared_error")
print(f"CV RMSE (log): {-scores.mean():.4f}")
```

## Step 4 — Final Prediction

```python
model.fit(X, y)
preds_log = model.predict(X_test)
preds = np.expm1(preds_log)             # back to dollars
```

Full notebook: `notebooks/house_price_pipeline.ipynb`
