# Implementation — Loan Default Prediction

## 1. Feature Engineering & Preprocessing

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

df = pd.read_parquet("data/lendingclub_clean.parquet")

# Feature engineering
df["credit_history_years"] = (
    pd.Timestamp("2020-01-01") - pd.to_datetime(df["earliest_cr_line"], errors="coerce")
).dt.days / 365.25
df["loan_inc_ratio"] = df["loan_amnt"] / (df["annual_inc"] + 1)
df["delinq_flag"] = (df["delinq_2yrs"] > 0).astype(int)

numeric = ["loan_amnt", "int_rate", "annual_inc", "dti", "revol_util",
           "credit_history_years", "loan_inc_ratio", "delinq_2yrs"]
categorical = ["grade", "home_ownership", "term"]
```

## 2. Pipeline & Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical)
])

model = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42))
])

X = df[numeric + categorical]
y = df["default"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
model.fit(X_train, y_train)
```

## 3. SHAP Explanation

```python
import shap

explainer = shap.LinearExplainer(model.named_steps["clf"],
                                 model.named_steps["prep"].transform(X_train[:1000]))
X_test_proc = model.named_steps["prep"].transform(X_test)
shap_values = explainer.shap_values(X_test_proc)

# Per-row explanation for a single applicant
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0, :],
                feature_names=model.named_steps["prep"].get_feature_names_out())
```

**Key decisions:**
- `C=0.1` — stronger L2 regularization reduces overfit on 100+ features.
- `class_weight="balanced"` handles 78/22 split without SMOTE.
- SHAP LinearExplainer gives exact feature contributions for logistic regression.
