# Support Vector Classifier (SVC)

Finds the **hyperplane** that best separates classes by maximizing the **margin** between them.

## How It Works

SVC finds the decision boundary that:
1. Correctly separates classes (or minimizes misclassification)
2. Maximizes the distance to the nearest points from each class (the **support vectors**)

Support vectors are the training points closest to the decision boundary. If you remove any non-support-vector point, the boundary doesn't change.

## Kernel Trick

SVC can create non-linear decision boundaries using the **kernel trick** — implicitly map data to a higher-dimensional space without computing the mapping explicitly.

| Kernel | When to Use |
|---|---|
| **Linear** | Data is approximately linearly separable. Fast, interpretable. |
| **RBF** (default) | Non-linear relationships. Most common. One hyperparameter (γ) controls flexibility. |
| **Polynomial** | Data has polynomial decision boundary. Less common. |
| **Sigmoid** | Similar to a neural network. Rarely used. |

## Key Hyperparameters

| Parameter | Effect |
|---|---|
| **C** | Regularization. High C = narrow margin (hard margin, overfit risk). Low C = wider margin (soft margin, better generalization). |
| **γ (gamma)** | RBF kernel influence. High γ = each point has small influence (complex boundary, overfit). Low γ = each point influences wide area (smooth boundary). |

## When to Use

- Small to medium datasets (<10K samples)
- High-dimensional data (features > samples, e.g., genomics)
- Clear margin separation expected
- Need for a powerful non-linear model without neural network complexity

## Examples

1. **Image classification (small data)**: 2K labeled images of cats vs dogs, each represented as 4096 features (deep features from pretrained CNN). SVC with RBF kernel achieves state-of-the-art for the dataset size.
2. **Text classification (high-dim)**: 10K TF-IDF features from news articles. SVC with linear kernel is fast and effective for topic classification — one of the best algorithms for high-dimensional sparse text data.
3. **Handwriting recognition**: MNIST digits (8×8 pixels = 64 features). SVC with RBF kernel achieves ~98% accuracy without any feature engineering.
