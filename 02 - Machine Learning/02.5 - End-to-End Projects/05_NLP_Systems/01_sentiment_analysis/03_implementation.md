# Sentiment Analysis — Implementation

## Pipeline

```
Raw text → Clean (lowercase, remove HTML/numbers) → TF-IDF vectorise → Train classifier → Evaluate
Compare unigrams vs unigrams+bigrams, LogisticRegression vs MultinomialNB
```

## Key Steps

### 1. Text cleaning and TF-IDF vectorisation

```python
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)     # strip HTML
    text = re.sub(r'\d+', '', text)          # remove numbers
    text = text.lower()
    return text

docs_clean = [clean_text(d) for d in raw_docs]

# Unigram-only vectoriser
vec_uni = TfidfVectorizer(max_features=50000, stop_words='english')
X_uni = vec_uni.fit_transform(docs_clean)

# Unigram + bigram vectoriser
vec_bi = TfidfVectorizer(max_features=100000, ngram_range=(1, 2),
                          stop_words='english', sublinear_tf=True)
X_bi = vec_bi.fit_transform(docs_clean)
```

### 2. Train Logistic Regression and MultinomialNB

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_bi, y, test_size=0.2, random_state=42
)

lr = LogisticRegression(C=1.0, max_iter=500, solver='saga', n_jobs=-1)
lr.fit(X_train, y_train)

nb = MultinomialNB(alpha=0.1)
nb.fit(X_train, y_train)

print(f"LR  accuracy: {lr.score(X_test, y_test):.4f}")
print(f"MNB accuracy: {nb.score(X_test, y_test):.4f}")
```

### 3. ROC AUC evaluation

```python
from sklearn.metrics import roc_auc_score, roc_curve

y_prob_lr = lr.predict_proba(X_test)[:, 1]
y_prob_nb = nb.predict_proba(X_test)[:, 1]

print(f"LR  AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")
print(f"MNB AUC: {roc_auc_score(y_test, y_prob_nb):.4f}")
```
