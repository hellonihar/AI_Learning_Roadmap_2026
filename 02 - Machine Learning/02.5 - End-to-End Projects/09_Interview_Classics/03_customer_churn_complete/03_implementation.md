# Implementation — Customer Churn

## Pipeline

1. Preprocessing: sklearn `ColumnTransformer` + `Pipeline`
2. Gradient Boosting with cross-validation
3. Cost-benefit analysis
4. Experiment tracking

## Code

### 1. Full sklearn pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df["Churn"] = (df["Churn"] == "Yes").astype(int)

X = df.drop(["customerID", "Churn"], axis=1)
y = df["Churn"]

# --- Feature engineering ---
X["TenureGroup"] = pd.cut(X["tenure"], bins=[0, 6, 12, 24, 48, 72],
                          labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
X["AvgMonthlyCharge"] = X["TotalCharges"] / (X["tenure"] + 1)
X["HasOnlineSecurity"] = (X["OnlineSecurity"] == "Yes").astype(int)
X["HasTechSupport"] = (X["TechSupport"] == "Yes").astype(int)
X["IsMonthToMonth"] = (X["Contract"] == "Month-to-month").astype(int)

X.drop(["OnlineSecurity", "TechSupport", "Contract"], axis=1, inplace=True)

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols)
])

pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    ))
])

scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
print(f"CV AUC-ROC: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 2. Cost-benefit analysis

```python
pipeline.fit(X, y)
y_prob = pipeline.predict_proba(X)[:, 1]

clv = 500
offer_cost = 50
retention_rate = 0.10

threshold = np.percentile(y_prob, 80)
targeted = y_prob >= threshold

n_targeted = targeted.sum()
n_churn_in_target = y[targeted].sum()
n_retained = n_churn_in_target * retention_rate

cost = n_targeted * offer_cost
benefit = n_retained * clv
net = benefit - cost

print(f"Targeted: {n_targeted}")
print(f"Retained: {n_retained:.0f}")
print(f"Cost: ${cost:,.0f}  Benefit: ${benefit:,.0f}  Net: ${net:,.0f}")
```

### 3. Experiment tracking

```python
# Pseudo-MLflow
experiment = {
    "model": "GradientBoosting",
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "features": X.shape[1],
    "cv_auc": scores.mean(),
    "net_benefit": net
}
# In production: mlflow.log_params(experiment)
```
