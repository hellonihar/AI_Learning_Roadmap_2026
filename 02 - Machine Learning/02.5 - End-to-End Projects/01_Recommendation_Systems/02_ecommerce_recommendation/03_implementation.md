# Implementation — E‑Commerce Hybrid Recommender

## 1. Candidate Generation (iALS)

```python
from implicit.als import AlternatingLeastSquares
import scipy.sparse as sp

user_item = sp.csr_matrix((df['rating'].values,
                           (df['user_id'], df['item_id'])))
model = AlternatingLeastSquares(factors=64, iterations=15, regularization=0.01)
model.fit(user_item.T)

# Generate 100 candidates per user
candidates = {}
for uid in range(n_users):
    ids, scores = model.recommend(uid, user_item[uid], N=100)
    candidates[uid] = ids.tolist()
```

## 2. Feature Engineering for Ranking

```python
def build_features(user_id, item_id, df, meta):
    feats = {}
    feats['rating_count_user'] = df[df.user_id == user_id].shape[0]
    feats['rating_count_item'] = df[df.item_id == item_id].shape[0]
    feats['price'] = meta.loc[item_id, 'price']
    feats['cat_match'] = int(meta.loc[user_id, 'top_cat']
                             == meta.loc[item_id, 'category'])
    feats['brand_known'] = int(pd.notna(meta.loc[item_id, 'brand']))
    return feats
```

## 3. LightGBM Ranker

```python
import lightgbm as lgb

train_data = lgb.Dataset(X_train, y_train, group=qids_train)
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [5, 20],
    'learning_rate': 0.05,
    'num_leaves': 63
}
ranker = lgb.train(params, train_data, valid_sets=[val_data])

Recall@20 = evaluate_recall(ranker, X_test, y_test, qids_test, k=20)
```
