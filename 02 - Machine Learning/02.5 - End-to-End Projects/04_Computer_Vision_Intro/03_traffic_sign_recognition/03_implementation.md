# Traffic Sign Recognition — Implementation

## Pipeline

```
Load images → Resize → Feature extraction (HOG + color hist) → PCA → Scale → SVC → Evaluate
Data augmentation via affine transforms on training set
```

## Key Steps

### 1. Data augmentation with affine transforms

```python
from skimage import transform as tf
import numpy as np

def augment_image(img):
    t = tf.AffineTransform(
        scale=(0.9, 1.1),
        rotation=(-15, 15),        # degrees
        translation=(-2, 2),
        shear=(-5, 5)
    )
    return tf.warp(img, t, mode='edge')

# Generate 2 augmented copies per training sample
def augment_dataset(images, labels):
    aug_imgs, aug_lbls = [], []
    for img, lbl in zip(images, labels):
        aug_imgs.append(img)
        aug_lbls.append(lbl)
        for _ in range(2):
            aug_imgs.append(augment_image(img))
            aug_lbls.append(lbl)
    return np.array(aug_imgs), np.array(aug_lbls)
```

### 2. HOG + color feature extraction

```python
from skimage.feature import hog
from skimage.color import rgb2gray

def extract_hog_color(images_rgb):
    features = []
    for img in images_rgb:
        gray = rgb2gray(img)
        hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), feature_vector=True)
        r_hist = np.histogram(img[:, :, 0], bins=16, range=(0, 1))[0]
        g_hist = np.histogram(img[:, :, 1], bins=16, range=(0, 1))[0]
        b_hist = np.histogram(img[:, :, 2], bins=16, range=(0, 1))[0]
        features.append(np.r_[hog_feat, r_hist, g_hist, b_hist])
    return np.array(features)
```

### 3. PCA + SVC

```python
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# X_feat shape ~117k × 624 (after augmentation)
pca = PCA(n_components=0.95)  # retain 95% variance
svm = SVC(kernel='rbf', gamma='scale', C=10, class_weight='balanced')

pipe = make_pipeline(StandardScaler(), pca, svm)
pipe.fit(X_train_feat, y_train)
print(f"Test accuracy: {pipe.score(X_test_feat, y_test):.4f}")
```

### 4. Per-class evaluation

```python
from sklearn.metrics import classification_report
y_pred = pipe.predict(X_test_feat)
print(classification_report(y_test, y_pred, zero_division=0))
```
