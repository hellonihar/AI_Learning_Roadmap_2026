# Scikit-Learn Essentials

## Estimator API
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
preds = model.predict(X_test)
probs = model.predict_proba(X_test)
```

## Pipelines
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=20)),
    ("clf", LogisticRegression())
])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```

## ColumnTransformer
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(), categorical_cols)
])
```

## Metrics
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
```

## Cross-Validation
```python
from sklearn.model_selection import cross_val_score, GridSearchCV

scores = cross_val_score(model, X, y, cv=5)
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X, y)
grid.best_params_
```
