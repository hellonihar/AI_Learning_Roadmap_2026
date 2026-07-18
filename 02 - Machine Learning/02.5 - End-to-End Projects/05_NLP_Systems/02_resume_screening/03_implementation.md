# Resume Screening — Implementation

## Pipeline

```
Raw resume text → Clean → Rule-based skill extraction → TF-IDF vectorise → OneVsRest SVM → Evaluate
```

## Key Steps

### 1. Skill extraction via pattern matching

```python
import re

SKILL_PATTERNS = {
    'python': r'\b(python|py)\b',
    'machine_learning': r'\b(machine learning|ml|deep learning|neural)\b',
    'excel': r'\b(excel|spreadsheet)\b',
    'sql': r'\b(sql|mysql|postgresql|query)\b',
    'seo': r'\b(seo|search engine|google analytics)\b',
    'gaap': r'\b(gaap|financial reporting|audit)\b',
    'recruiting': r'\b(recruit|talent acquisition|sourcing)\b',
}

def extract_skills(text):
    text = text.lower()
    found = {}
    for skill, pattern in SKILL_PATTERNS.items():
        found[skill] = 1 if re.search(pattern, text) else 0
    return found

# Example: "Proficient in Python, SQL and ML" → {python:1, sql:1, machine_learning:1}
```

### 2. TF-IDF + OneVsRest SVM

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer

vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                       stop_words='english', sublinear_tf=True)
X = vec.fit_transform(resume_texts)

# y is list of lists, e.g. [['Data Science', 'Software Engineering'], ...]
mlb = MultiLabelBinarizer()
y_bin = mlb.fit_transform(y)

clf = OneVsRestClassifier(LinearSVC(C=1.0, class_weight='balanced', max_iter=2000))
clf.fit(X, y_bin)
print(f"Train accuracy: {clf.score(X, y_bin):.4f}")
```

### 3. Evaluation per class

```python
from sklearn.metrics import f1_score, classification_report

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=mlb.classes_))
```
