# Implementation — Spam Detection

## 1. Preprocessing & Feature Extraction

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/spam_combined.csv")
df["clean"] = df["text"].str.lower().str.replace(r"[^\w\s@.]", "", regex=True)

vectorizer = TfidfVectorizer(
    max_features=10_000, ngram_range=(1, 3),
    sublinear_tf=True, stop_words="english"
)
X = vectorizer.fit_transform(df["clean"])
y = df["label"]
```

## 2. Train / Validation Split & Model Training

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

base = MultinomialNB(alpha=0.1)
model = CalibratedClassifierCV(base, method="isotonic")
model.fit(X_train, y_train)
```

## 3. Evaluation — Precision-Focused Threshold

```python
from sklearn.metrics import precision_recall_curve, classification_report

probs = model.predict_proba(X_val)[:, 1]

# Find threshold that gives precision >= 0.98
precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
target_mask = precisions[:-1] >= 0.98
best_idx = target_mask.argmax()
threshold = thresholds[best_idx]

y_pred = (probs >= threshold).astype(int)
print(classification_report(y_val, y_pred))
```

**Key decisions:**
- `ngram_range=(1, 3)` — bigrams and trigrams catch phrases like "click here" or "free money".
- `sublinear_tf=True` — dampens term-frequency explosion in long emails.
- Isotonic calibration converts raw NB scores into well-calibrated probabilities for threshold tuning.
