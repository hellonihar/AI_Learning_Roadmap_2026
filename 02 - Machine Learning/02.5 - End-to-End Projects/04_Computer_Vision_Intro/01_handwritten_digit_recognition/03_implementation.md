# Handwritten Digit Recognition — Implementation

## Pipeline

```
Raw pixels → Feature extraction (HOG / raw pixels) → Train/Test split → Scale → Classifier → Evaluate
```

## Key Steps

### 1. Load and explore data

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.astype(np.float32)
y = mnist.target.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42
)
```

### 2. HOG feature extraction + SVC

```python
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def extract_hog(images):
    features = []
    for img in images.values.reshape(-1, 28, 28):
        fd = hog(img, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), feature_vector=True)
        features.append(fd)
    return np.array(features)

X_train_hog = extract_hog(X_train)
X_test_hog = extract_hog(X_test)

svm_clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='scale', C=1.0))
svm_clf.fit(X_train_hog, y_train)
print(f"SVM (HOG) accuracy: {svm_clf.score(X_test_hog, y_test):.4f}")
```

### 3. Raw pixels + Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
rf_clf.fit(X_train, y_train)
print(f"RF (pixels) accuracy: {rf_clf.score(X_test, y_test):.4f}")
```

### 4. Confusion matrix analysis

```python
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

y_pred = svm_clf.predict(X_test_hog)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.savefig('confusion_matrix_svm_hog.png')
print(classification_report(y_test, y_pred))
```
