# Implementation — News Feed CTR Ranking

## 1. Feature Engineering

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df['recency_min'] = (df['impression_time'] - df['publish_time']).dt.total_seconds() / 60
df['user_ctr'] = df.groupby('user_id')['click'].transform('mean')
df['article_pop_1h'] = df.groupby('article_id')['click'].transform(
    lambda x: x.rolling('1h', min_periods=1).mean())
df['category_match'] = (df['user_top_category'] == df['article_category']).astype(int)
```

## 2. Train XGBoost Classifier

```python
import xgboost as xgb

features = ['recency_min', 'user_ctr', 'article_pop_1h', 'category_match',
            'session_depth', 'device_enc', 'category_enc']
X, y = df[features], df['click']

model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    eval_metric='auc'
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```

## 3. NDCG@k Evaluation

```python
from sklearn.metrics import ndcg_score

def compute_ndcg(group):
    y_true = group['click'].values.reshape(1, -1)
    y_score = group['score'].values.reshape(1, -1)
    return ndcg_score(y_true, y_score, k=10)

df_test['score'] = model.predict_proba(df_test[features])[:, 1]
ndcg = df_test.groupby('impression_id').apply(compute_ndcg).mean()
print(f'NDCG@10: {ndcg:.4f}')
```
