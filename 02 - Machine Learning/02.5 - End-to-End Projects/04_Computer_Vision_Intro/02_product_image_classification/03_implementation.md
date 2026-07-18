# Product Image Classification — Implementation

## Pipeline

```
Load images → Color histogram + GLCM features → Concatenate → Scale → Logistic Regression → Evaluate
```

## Key Steps

### 1. Feature extraction

```python
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte

def extract_features(images):
    features = []
    for img in images.reshape(-1, 28, 28):
        img_u8 = img_as_ubyte(img / 255.0)
        hist = np.histogram(img_u8, bins=32, range=(0, 255))[0]
        glcm = graycomatrix(img_u8, distances=[1], angles=[0],
                            levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        features.append(np.r_[hist, contrast, homogeneity, energy, correlation])
    return np.array(features)
```

### 2. Train Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Assume X_raw is (60000, 784), y is (60000,)
X_feat = extract_features(X_raw)
X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y, test_size=10000, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000, C=1.0, multi_class='multinomial', solver='lbfgs')
clf.fit(X_train_scaled, y_train)
print(f"Accuracy: {clf.score(X_test_scaled, y_test):.4f}")
```

### 3. Baseline comparison (raw pixels)

```python
from sklearn.ensemble import RandomForestClassifier

rf_base = RandomForestClassifier(n_estimators=150, max_depth=15, n_jobs=-1, random_state=42)
rf_base.fit(X_train_raw, y_train)
print(f"Baseline RF (raw pixels): {rf_base.score(X_test_raw, y_test):.4f}")
```
