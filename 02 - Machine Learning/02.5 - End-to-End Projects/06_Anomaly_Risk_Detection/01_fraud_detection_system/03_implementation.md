# Fraud Detection — Implementation

## Steps

1. **EDA** — Distribution of TransactionAmt by hour, card frequency encoding, V* feature correlation analysis
2. **Feature Engineering** — Frequency encoding on card/addr/email columns, transaction hour/day-of-week, ratio features (TransactionAmt / card1 avg), aggregations per card
3. **Model Baseline** — Isolation Forest (unsupervised) to detect outliers, then supervised XGBoost
4. **Threshold Tuning** — Optimize decision threshold on validation PR curve
5. **Evaluation** — Stratified cross-validation using time-based splits

## Key Code

```python
# Isolation Forest baseline
from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    n_estimators=200, contamination=0.05, random_state=42,
    n_jobs=-1
)
iso_preds = iso.fit_predict(X_scaled)
# Convert: -1 = anomaly, 1 = normal -> fraud probability
anomaly_scores = -iso.score_samples(X_scaled)
```

```python
# XGBoost with scale_pos_weight
from xgboost import XGBClassifier

ratio = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    scale_pos_weight=ratio, subsample=0.8, colsample_bytree=0.6,
    eval_metric='aucpr', early_stopping_rounds=20
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```

```python
# Threshold tuning on PR curve
from sklearn.metrics import precision_recall_curve

probs = model.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, probs)

f_score = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_threshold = thresholds[np.argmax(f_score[:-1])]
print(f'Optimal threshold: {best_threshold:.4f}, F2: {f_score.max():.4f}')
```
