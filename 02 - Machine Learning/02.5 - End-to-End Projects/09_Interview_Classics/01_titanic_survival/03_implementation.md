# Implementation — Titanic Survival

## Full Feature Engineering + Modeling

### 1. Complete feature engineering

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("train.csv")

# --- Title from Name ---
title_map = {
    "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Don": "Rare", "Lady": "Rare", "Sir": "Rare", "Countess": "Rare",
    "Capt": "Rare", "Jonkheer": "Rare"
}
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.")[0].map(title_map)

# --- Family ---
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# --- Cabin deck ---
df["Deck"] = df["Cabin"].str[0].fillna("U")

# --- Age imputation by Title ---
age_medians = df.groupby("Title")["Age"].median().to_dict()
df["Age"] = df["Age"].fillna(df["Title"].map(age_medians))

# --- Ticket frequency ---
ticket_counts = df["Ticket"].value_counts()
df["TicketFreq"] = df["Ticket"].map(ticket_counts)
df["FarePerPerson"] = df["Fare"] / df["TicketFreq"]

# --- Embarked ---
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# --- Encode ---
df["Sex"] = (df["Sex"] == "male").astype(int)
df = pd.get_dummies(df, columns=["Title", "Deck", "Embarked"], drop_first=True)

feature_cols = [c for c in df.columns if c not in [
    "PassengerId", "Survived", "Name", "Ticket", "Cabin"
]]
X = df[feature_cols]
y = df["Survived"]
```

### 2. Model training with cross-validation

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

models = {
    "RF": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "GBM": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"{name}: Acc={scores.mean():.4f} (+/-{scores.std():.4f})  AUC={aucs.mean():.4f}")
```

### 3. Final model and test prediction

```python
final_model = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42
)
final_model.fit(X, y)

# Predict test set (same feature engineering on test)
test = pd.read_csv("test.csv")
# ... apply same feature engineering ...
test_preds = final_model.predict(test[feature_cols])

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_preds
})
submission.to_csv("submission.csv", index=False)
```
