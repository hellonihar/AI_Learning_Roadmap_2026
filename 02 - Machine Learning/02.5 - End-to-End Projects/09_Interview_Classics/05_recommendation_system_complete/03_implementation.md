# Implementation — Recommendation System

## Pipeline

1. SVD matrix factorization (surprise library)
2. Cold-start fallback (popular items)
3. REST API design
4. A/B test framework

## Code

### 1. SVD with hyperparameter tuning

```python
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV, cross_validate

reader = Reader(line_format="user item rating timestamp", sep="\t", rating_scale=(1, 5))
data = Dataset.load_from_file("u.data", reader=reader)

# Grid search
param_grid = {
    "n_factors": [50, 100, 150],
    "lr_all": [0.002, 0.005],
    "reg_all": [0.02, 0.1]
}
gs = GridSearchCV(SVD, param_grid, cv=3, n_jobs=-1, measures=["rmse"])
gs.fit(data)

print(f"Best params: {gs.best_params}")
print(f"Best RMSE: {gs.best_score['rmse']:.4f}")

best_svd = gs.best_estimator["rmse"]
trainset = data.build_full_trainset()
best_svd.fit(trainset)
```

### 2. Cold-start handling

```python
# Popular items (weighted by rating count)
ratings = pd.read_csv("u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
item_stats = ratings.groupby("item_id")["rating"].agg(["mean", "count"])
item_stats["score"] = item_stats["mean"] * np.log1p(item_stats["count"])
popular_items = item_stats.sort_values("score", ascending=False).head(20).index.tolist()

def recommend(user_id, n=10):
    if user_id not in trainset.all_users():
        return popular_items[:n]

    rated = set([j for (j, _) in trainset.ur[user_id]])
    candidates = [iid for iid in trainset.all_items() if iid not in rated]
    predictions = [(iid, best_svd.predict(user_id, iid).est) for iid in candidates]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [iid for iid, _ in predictions[:n]]
```

### 3. REST API

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class RecRequest(BaseModel):
    user_id: int
    n: int = 10

@app.post("/recommend")
def recommend(req: RecRequest):
    items = get_recommendations(req.user_id, req.n)
    return {"user_id": req.user_id, "recommendations": items}

@app.post("/predict-rating")
def predict_rating(user_id: int, item_id: int):
    if user_id in trainset.all_users() and item_id in trainset.all_items():
        pred = best_svd.predict(user_id, item_id).est
    else:
        pred = item_stats.loc[item_id, "mean"] if item_id in item_stats.index else 3.0
    return {"user_id": user_id, "item_id": item_id, "predicted_rating": round(pred, 2)}
```

### 4. A/B test framework

```python
# A/B test: SVD vs. Popularity
# Randomly assign users to control (popularity) or treatment (SVD)
# Metrics: CTR, avg rating of consumed items, 7-day retention

def run_ab_test(users, model_a, model_b, n=10):
    results = []
    for user in users:
        recs_a = model_a.recommend(user, n)
        recs_b = model_b.recommend(user, n)
        results.append({
            "user": user,
            "model_a_recs": recs_a,
            "model_b_recs": recs_b,
            "overlap": len(set(recs_a) & set(recs_b))
        })
    return pd.DataFrame(results)

# In production: random 50/50 split, measure for 2 weeks
# Metric: CTR = clicks / impressions
```
