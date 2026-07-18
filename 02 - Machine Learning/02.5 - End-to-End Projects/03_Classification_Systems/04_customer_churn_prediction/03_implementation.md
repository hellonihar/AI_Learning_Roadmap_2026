# Implementation — Customer Churn Prediction

## 1. Feature Engineering Deep-Dive

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data/telcom_churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

df["avg_monthly_charge"] = df["TotalCharges"] / (df["tenure"] + 1)
df["service_count"] = df[[c for c in df.columns if c.startswith("Online")
                          or c.startswith("Streaming") or c in [
    "PhoneService", "MultipleLines", "TechSupport", "DeviceProtection"
]]].apply(lambda row: (row == "Yes").sum(), axis=1)

df["tenure_segment"] = pd.cut(df["tenure"], bins=[0, 6, 12, 24, 48, 100],
                               labels=["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4yr+"])
df["charge_trend"] = df.groupby("customerID")["MonthlyCharges"].transform(
    lambda x: x.pct_change().rolling(3).mean()
)
```

## 2. Model Training — Random Forest & XGBoost

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

cats = ["gender", "Contract", "InternetService", "PaymentMethod", "tenure_segment"]
X = pd.get_dummies(df.drop(["customerID", "Churn"], axis=1), columns=cats)
y = df["Churn"].map({"Yes": 1, "No": 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=20,
                            class_weight="balanced_subsample", random_state=42, n_jobs=-1)
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                    eval_metric="logloss", random_state=42)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
```

## 3. Evaluation with Recall Focus

```python
from sklearn.metrics import classification_report, f1_score

rf_probs = rf.predict_proba(X_test)[:, 1]
xgb_probs = xgb.predict_proba(X_test)[:, 1]

# Tune threshold for recall >= 0.85
for model_name, probs in [("RF", rf_probs), ("XGB", xgb_probs)]:
    for thr in [0.3, 0.4, 0.5]:
        preds = (probs >= thr).astype(int)
        report = classification_report(y_test, preds, output_dict=True)
        print(f"{model_name} @ {thr}: R={report['1']['recall']:.3f} "
              f"P={report['1']['precision']:.3f} F1={f1_score(y_test, preds):.3f}")
```

**Key decisions:**
- `class_weight="balanced_subsample"` in RF — subsample weights handle churn imbalance.
- `scale_pos_weight` in XGBoost directly mirrors the imbalance ratio.
- Threshold tuning at inference time, not training time.
