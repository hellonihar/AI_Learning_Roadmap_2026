# Implementation — Movie Recommendation with SVD

## 1. Preprocessing

```python
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

df = pd.read_csv('ml-100k/u.data', sep='\t',
                 names=['userId', 'movieId', 'rating', 'timestamp'])
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
```

## 2. Train SVD Model

```python
model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
model.fit(trainset)

predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse:.4f}')
```

## 3. Precision@k Evaluation

```python
from collections import defaultdict

def precision_at_k(predictions, k=10, threshold=4.0):
    user_est = defaultdict(list)
    for uid, iid, true, est, _ in predictions:
        user_est[uid].append((est, true))
    precisions = []
    for uid in user_est:
        user_est[uid].sort(key=lambda x: x[0], reverse=True)
        top_k = user_est[uid][:k]
        n_hit = sum(1 for _, true in top_k if true >= threshold)
        precisions.append(n_hit / k)
    return np.mean(precisions)

p10 = precision_at_k(predictions, k=10, threshold=4.0)
print(f'Precision@10: {p10:.3f}')
```
