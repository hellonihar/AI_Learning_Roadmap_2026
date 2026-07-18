# Handwritten Digit Recognition — Dataset

## Source
[Modified National Institute of Standards and Technology (MNIST)](http://yann.lecun.com/exdb/mnist/). Loaded via `sklearn.datasets.fetch_openml('mnist_784')`.

## Size
| Split | Samples |
|-------|---------|
| Train | 60,000 |
| Test  | 10,000 |

## Features
- **Pixel features**: 784 raw pixel values (28×28 image, 0–255 greyscale)
- **HOG features**: 324-dimensional vector (9 orientations, 8×8 cells, 2×2 blocks), computed via `skimage.feature.hog`
- **Target**: Integer label 0–9

## Challenges
- Handwriting variability across writers
- Similar digit shapes (3 vs 8, 4 vs 9)
- Thin strokes vs bold strokes affect pixel distributions
