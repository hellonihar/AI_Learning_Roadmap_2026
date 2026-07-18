# Product Image Classification — Dataset

## Source
[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) — 70,000 greyscale 28×28 images of clothing items. Loaded via `keras.datasets.fashion_mnist` or downloaded as raw CSV.

## Size
| Split | Samples |
|-------|---------|
| Train | 60,000 |
| Test  | 10,000 |

## Classes
0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

## Features
- **Color histogram**: 32-bin histogram per image (greyscale, so 1 channel × 32 bins = 32-d)
- **Texture**: GLCM contrast + homogeneity + energy + correlation (4-d) via `skimage.feature.greycomatrix`
- **Raw pixels** (784-d) as baseline

Total engineered feature vector: 36-d

## Challenges
- Shirt vs T-shirt vs Pullover are visually similar (fine-grained)
- Lighting / contrast variation in real-world settings (not in Fashion MNIST — add noise for robustness)
- Scale invariance needed (Fashion MNIST is pre-centered)
