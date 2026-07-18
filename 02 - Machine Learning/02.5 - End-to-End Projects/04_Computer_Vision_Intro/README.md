# Computer Vision Intro — End-to-End Projects

Classical CV pipeline projects using scikit-learn. No deep learning.

| Project | Dataset | Approach | Key Concept |
|---------|---------|----------|-------------|
| 1. Handwritten Digit Recognition | MNIST | HOG + SVC, Pixel RF | Feature engineering |
| 2. Product Image Classification | Fashion MNIST | Color histograms + Texture + LR | Feature extraction |
| 3. Traffic Sign Recognition | GTSRB | HOG + Color + PCA + SVC | Data augmentation |

### Setup

```bash
pip install scikit-learn numpy pandas matplotlib opencv-python scikit-image
```

### How to Run

Each project is self-contained. Navigate to the project folder and run the implementation notebook/script.

### Learning Path

1. Start with handwritten digit recognition (simplest features)
2. Move to product classification (multiple feature types)
3. End with traffic signs (augmentation + PCA)
