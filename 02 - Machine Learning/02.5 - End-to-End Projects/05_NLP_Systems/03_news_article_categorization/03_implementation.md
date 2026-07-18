# News Article Categorisation — Implementation

## Pipeline

```
Load 20 Newsgroups → Clean (headers/footers/quotes) → TF-IDF vectorise → Linear SVC → Evaluate → LDA topic modelling
```

## Key Steps

### 1. Load and clean

```python
from sklearn.datasets import fetch_20newsgroups

# Remove headers, footers, and quotes to get clean body text
categories = None  # use all 20
news_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
news_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

print(f"Train: {len(news_train.data)}, Test: {len(news_test.data)}")
print(f"Classes: {len(news_train.target_names)}")
```

### 2. TF-IDF + Linear SVC

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                       stop_words='english', sublinear_tf=True)
clf = LinearSVC(C=1.0, max_iter=5000, dual=False)

pipe = make_pipeline(vec, clf)
pipe.fit(news_train.data, news_train.target)

acc = pipe.score(news_test.data, news_test.target)
print(f"Test accuracy: {acc:.4f}")
```

### 3. LDA topic exploration

```python
from sklearn.decomposition import LatentDirichletAllocation

# Fit LDA on the TF-IDF matrix
X_vec = vec.fit_transform(news_train.data)
lda = LatentDirichletAllocation(n_components=20, max_iter=10, random_state=42)
lda.fit(X_vec)

# Show top words per topic
feature_names = vec.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")
```

### 4. Per-class evaluation

```python
from sklearn.metrics import classification_report
y_pred = pipe.predict(news_test.data)
print(classification_report(news_test.target, y_pred, target_names=news_test.target_names))
```
