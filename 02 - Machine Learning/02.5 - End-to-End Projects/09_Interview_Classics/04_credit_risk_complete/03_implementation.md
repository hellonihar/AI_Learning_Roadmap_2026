# Implementation — Credit Risk

## Pipeline

1. WoE binning for each feature (monotonic bins)
2. Logistic Regression on WoE-transformed features
3. Scorecard conversion (points per feature level)
4. SHAP explanation
5. Regulatory documentation

## Code

### 1. WoE binning

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import shap

df = pd.read_csv("german_credit.csv")
# Map target: 1=good -> 0, 2=bad -> 1
df["risk"] = (df["risk"] == 2).astype(int)

def woe_binning(series, target, bins=5):
    """Calculate WoE and IV for a numeric feature."""
    if series.nunique() <= bins:
        groups = series
    else:
        groups = pd.qcut(series, q=bins, duplicates="drop")
    grouped = pd.DataFrame({"feature": groups, "target": target})
    stats = grouped.groupby("feature").agg(
        total=("target", "count"),
        bad=("target", "sum")
    ).reset_index()
    stats["good"] = stats["total"] - stats["bad"]
    stats["bad_rate"] = stats["bad"] / stats["bad"].sum()
    stats["good_rate"] = stats["good"] / stats["good"].sum()
    stats["WoE"] = np.log(stats["good_rate"] / stats["bad_rate"].replace(0, 1e-6))
    stats["IV"] = (stats["good_rate"] - stats["bad_rate"]) * stats["WoE"]
    return stats

# Apply WoE to each feature
for col in ["age", "credit_amount", "duration"]:
    woe_df = woe_binning(df[col], df["risk"])
    bin_map = dict(zip(woe_df["feature"], woe_df["WoE"]))
    df[f"{col}_WoE"] = df[col].apply(
        lambda x: next(v for k, v in bin_map.items() if x in k)
    )
```

### 2. Logistic regression on WoE features

```python
woe_cols = [c for c in df.columns if c.endswith("_WoE")]
X_woe = df[woe_cols]
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X_woe, y, test_size=0.3, random_state=42, stratify=y
)

lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

y_prob = lr.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"AUC-ROC: {auc:.4f}")
```

### 2. Scorecard conversion

```python
# Score = Offset + Factor * log(odds)
# Set: score 600 for odds 50:1, double odds → +20 points
factor = 20 / np.log(2)
offset = 600 - factor * np.log(50)

# Points per feature level
for i, col in enumerate(woe_cols):
    coef = lr.coef_[0][i]
    points = round(-coef * factor)  # negative because higher WoE = lower risk
    print(f"{col}: {points} points per WoE unit")

# Score = offset + sum(points * WoE)
# Higher score = lower risk
```

### 2. SHAP explanation

```python
import shap

explainer = shap.LinearExplainer(lr, X_train)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=woe_cols)
# Top drivers: credit_amount, duration, checking_account
```
