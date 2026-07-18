# Linear Discriminant Analysis (LDA)

A **supervised** dimensionality reduction technique that finds components maximizing **class separability**.

## How LDA Works

LDA finds the projection that:
1. **Maximizes** the distance between class means (separability)
2. **Minimizes** the variance within each class (compactness)

Unlike PCA (which ignores class labels), LDA uses label information to find the most discriminative directions.

## LDA vs PCA

| Aspect | PCA | LDA |
|---|---|---|
| **Supervised?** | No | Yes |
| **Goal** | Maximize variance | Maximize class separation |
| **Max components** | n_features | n_classes — 1 |
| **Best for** | Unsupervised, reconstruction | Classification preprocessing |

## When to Use

- Classification with high-dimensional data
- Visualization of class separation (reduce to 2D-3D)
- Feature extraction before a simpler classifier (e.g., LDA + kNN)
- Normally distributed features with equal covariance per class

## Limitations

- Assumes **normally distributed** features with **equal covariance** per class
- Can only produce up to **k-1 components** (2 classes → 1 component)
- Doesn't work well for non-linear boundaries (try Kernel LDA)

## Examples

1. **Face recognition (Fisherfaces)**: 40 people, 10 images each, 10K pixels. LDA finds components that best separate individuals. Compared to PCA (Eigenfaces, finds human-face-likeness), LDA (Fisherfaces, finds what distinguishes each person) achieves better recognition accuracy.
2. **Medical diagnosis**: 200 features (lab tests), 3 classes (healthy, disease A, disease B). LDA reduces to 2 components (k-1) that best separate the three groups. Visualize patient clusters in 2D.
3. **Credit scoring**: 50 features, 2 classes (default / no default). LDA finds the single linear combination that best separates defaulters from non-defaulters — essentially learning a custom credit score formula.
