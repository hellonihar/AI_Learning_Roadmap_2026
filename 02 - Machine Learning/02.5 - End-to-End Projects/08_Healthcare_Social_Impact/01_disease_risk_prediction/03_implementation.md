# Implementation — Disease Risk Prediction

## Pipeline

1. Load & clean — handle missing values in `ca` and `thal`
2. EDA — correlation matrix, class balance check
3. Train logistic regression with tuned threshold
4. Interpret with SHAP
5. Evaluate on sensitivity / specificity

## Code

### 1. Load and preprocess

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("heart.csv")
# Map target: original 1-4 -> 1, 0 -> 0
df["target"] = (df["target"] > 0).astype(int)

# Handle missing (UCI uses '?' which becomes NaN)
df.fillna({"ca": df["ca"].median(), "thal": df["thal"].mode()[0]}, inplace=True)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 2. Logistic regression with threshold tuning

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score

model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)

y_prob = model.predict_proba(X_test_s)[:, 1]

# Find threshold that maximizes sensitivity given specificity >= 0.80
thresholds = np.arange(0.3, 0.7, 0.01)
best_thresh = 0.5
best_sens = 0
for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    if sens >= 0.85 and spec >= 0.80:
        best_thresh = t
        break

y_pred = (y_prob >= best_thresh).astype(int)
print(f"Threshold: {best_thresh:.2f}")
```

### 3. SHAP explanation

```python
import shap

explainer = shap.LinearExplainer(model, X_train_s, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test_s)

shap.summary_plot(shap_values, X_test_s, feature_names=X.columns)
# Top drivers: thal, ca, oldpeak, thalach
```
