# Text Similarity / Duplicate Detection — Implementation

## Pipeline

```
Question pair → TF-IDF cosine + Jaccard + WMD + basic features → Feature vector (6-d) → Logistic Regression → AUC
```

## Key Steps

### 1. TF-IDF cosine and Jaccard

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fit TF-IDF on all unique questions
all_questions = list(set(q1s + q2s))
vec = TfidfVectorizer(max_features=50000, stop_words='english', sublinear_tf=True)
vec.fit(all_questions)

# Transform pair
q1_vec = vec.transform(q1s)
q2_vec = vec.transform(q2s)
cos_sim = cosine_similarity(q1_vec, q2_vec).diagonal()

def jaccard_similarity(s1, s2):
    set1, set2 = set(s1.split()), set(s2.split())
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0.0

jacc_sim = np.array([jaccard_similarity(a, b) for a, b in zip(q1s, q2s)])
```

### 2. Word Mover's Distance with pre-trained embeddings

```python
import gensim.downloader as api
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Load pre-trained word2vec (first time downloads ~1.6 GB)
w2v = api.load('word2vec-google-news-300')

def wmd(s1, s2, model):
    """Word Mover's Distance via emd"""
    from scipy.spatial.distance import cdist
    tokens1 = [w for w in s1.lower().split() if w in model]
    tokens2 = [w for w in s2.lower().split() if w in model]
    if not tokens1 or not tokens2:
        return 1.0
    vec1 = np.array([model[w] for w in tokens1])
    vec2 = np.array([model[w] for w in tokens2])
    dist_matrix = cdist(vec1, vec2, metric='euclidean')
    from scipy.stats import wasserstein_distance
    # Uniform weights
    w1 = np.ones(len(tokens1)) / len(tokens1)
    w2 = np.ones(len(tokens2)) / len(tokens2)
    # Relaxed WMD using wasserstein_distance for 1D approximation
    # Full WMD via optimal transport — use pyemd for production
    return float(dist_matrix.min(axis=1).sum() / len(tokens1))

# Compute WMD for a sample (full dataset ≈ 30 min)
wmd_scores = np.array([wmd(a, b, w2v) for a, b in zip(q1s[:5000], q2s[:5000])])
```

### 3. Combine features + Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Build feature matrix: [cos_sim, jacc_sim, wmd_sim, len_diff, common_words, longest_common_substr_ratio]
features = np.column_stack([
    cos_sim, jacc_sim, wmd_scores,
    np.abs([len(a.split()) - len(b.split()) for a, b in zip(q1s, q2s)]),
    [len(set(a.split()) & set(b.split())) for a, b in zip(q1s, q2s)],
    [len_common_substr(a, b) / max(len(a), len(b)) for a, b in zip(q1s, q2s)]
])

X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, random_state=42
)

lr = LogisticRegression(C=1.0, max_iter=500)
lr.fit(X_train, y_train)
print(f"AUC: {roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]):.4f}")
```
