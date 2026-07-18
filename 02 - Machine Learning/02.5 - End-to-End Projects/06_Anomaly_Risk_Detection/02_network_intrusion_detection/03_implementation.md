# Network Intrusion Detection — Implementation

## Steps

1. **Preprocessing** — One-hot encode protocol_type/service/flag, log-transform skewed features (src_bytes, dst_bytes), drop zero-variance columns
2. **Dimensionality Reduction** — PCA to 20 components for unsupervised methods
3. **Isolation Forest** — Train on full data (normal + attack) to isolate anomalies
4. **One-Class SVM** — Train on normal traffic only, flag deviations
5. **Ensemble** — Average anomaly scores from both models
6. **ROC Analysis** — Evaluate on known vs. novel attack categories separately

## Key Code

```python
# Preprocessing pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

cat_cols = ['protocol_type', 'service', 'flag']
num_cols = [c for c in df.columns if c not in cat_cols + ['label']]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])
X_scaled = preprocessor.fit_transform(X)
```

```python
# Isolation Forest
from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    n_estimators=300, contamination='auto',
    max_samples='auto', random_state=42
)
iso_scores = -iso.score_samples(X_scaled)

# One-Class SVM
from sklearn.svm import OneClassSVM

ocsvm = OneClassSVM(nu=0.01, kernel='rbf', gamma='scale')
ocsvm_scores = -ocsvm.score_samples(X_scaled)
```

```python
# ROC analysis per attack category
from sklearn.metrics import roc_auc_score, roc_curve

for attack_cat in ['dos', 'probe', 'r2l', 'u2r']:
    y_binary = (y_test == attack_cat).astype(int)
    # Skip if no positive samples
    if y_binary.sum() == 0:
        continue
    auc = roc_auc_score(y_binary, ensemble_scores)
    fpr, tpr, _ = roc_curve(y_binary, ensemble_scores)
    print(f'{attack_cat.upper():6s} | ROC AUC: {auc:.3f} | TPR @ 1% FPR: {tpr[np.argmax(fpr >= 0.01)]:.3f}')
```
