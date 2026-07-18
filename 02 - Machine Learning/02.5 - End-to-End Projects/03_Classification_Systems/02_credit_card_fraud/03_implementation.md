# Implementation — Credit Card Fraud Detection

## 1. Preprocessing & Resampling (SMOTE)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv("data/creditcard.csv")
df = df.drop("Time", axis=1)  # avoid temporal leakage
df["Amount"] = StandardScaler().fit_transform(df[["Amount"]])

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

smote = SMOTE(sampling_strategy=0.1, random_state=42)  # 10:1 legit:fraud
X_res, y_res = smote.fit_resample(X_train, y_train)
```

## 2. Model Training — LightGBM with Scale Pos Weight

```python
import lightgbm as lgb

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # ~580

model = lgb.LGBMClassifier(
    scale_pos_weight=scale_pos_weight,
    n_estimators=300, max_depth=6,
    learning_rate=0.05, random_state=42
)
model.fit(X_res, y_res)
```

## 3. Threshold Optimization via Precision-Recall Curve

```python
from sklearn.metrics import precision_recall_curve

probs = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

# Maximize F2 (recall-weighted)
f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-9)
best_idx = f2_scores.argmax()
best_threshold = thresholds[best_idx]

y_pred = (probs >= best_threshold).astype(int)
print(f"Best threshold: {best_threshold:.3f}, PR-AUC: {average_precision_score(y_test, probs):.4f}")
```

**Key decisions:**
- SMOTE with 10:1 ratio (not 1:1) — avoids over-sampling noise.
- `scale_pos_weight` inside LightGBM for dual handling of imbalance.
- F2-optimal threshold selected post-hoc — separates model training from decision boundary.
