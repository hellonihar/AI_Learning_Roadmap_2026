# Transfer Learning

## Concept

Transfer learning reuses a model trained on one task (typically a large dataset like ImageNet — 1.2M images, 1000 classes) as the starting point for a different but related task. This avoids training from scratch, which requires massive data and computation.

### Why It Works

CNN feature hierarchies are shared across visual tasks:

- **Early layers** detect generic features (edges, colors, blobs) — useful for any image task.
- **Middle layers** detect textures and patterns — somewhat task-specific.
- **Late layers** detect object parts and full objects — highly task-specific.

Since most vision tasks involve similar low-level visual patterns, early and middle layers transfer effectively.

## Strategies

### 1. Feature Extraction (Fixed Features)

The pre-trained model's convolutional base is used as a fixed feature extractor. No weights are updated during training on the new task.

```
Pre-trained Base (frozen) → New Classifier (trained from scratch)

Example: ResNet-50 (frozen) → GlobalAvgPool → Dense(256, ReLU) → Dense(num_classes, Softmax)
```

**Process:**
1. Load pre-trained model without its classification head.
2. Freeze all convolutional layers (`layer.trainable = False`).
3. Run all training images through the base once (cache the features).
4. Train only the newly added classifier layers.

**When to use:**
- Small target dataset (fewer than ~1000 images per class).
- Target dataset is similar to the pre-training dataset (e.g., ImageNet → general object classification).
- Limited computational resources.

### 2. Fine-Tuning

All layers (or a subset) are updated during training on the new task. The pre-trained weights serve as initialization rather than fixed features.

```
Pre-trained Base (unfrozen, low LR) → New Classifier (trained)

Learning rate: typically 1/10th of the default (e.g., 1e-4 vs 1e-3).
```

**Process:**
1. Load pre-trained model, replace classifier head.
2. Optionally freeze base, train classifier for a few epochs (warm-up).
3. Unfreeze the base (or selected layers).
4. Continue training end-to-end with a low learning rate.

**When to use:**
- Large target dataset (thousands of images per class).
- Target dataset is significantly different from pre-training data (e.g., medical images, satellite imagery).
- You have enough compute to backpropagate through the entire network.

### 3. Freezing Layers

Not all layers need to be unfrozen. A common approach:

```
Scenario: Target task is similar but dataset is medium-sized.
  Option A: Freeze early layers, fine-tune later layers.
  Option B: Fine-tune all layers with a very low LR.
```

**Freezing rules of thumb:**

| Dataset Size | Similar to Pre-training | Different from Pre-training |
|---|---|---|
| Small | Feature extraction | Fine-tune last 1/3 of layers |
| Medium | Fine-tune last 1/2 of layers | Fine-tune all (low LR) |
| Large | Fine-tune all (standard LR) | Fine-tune all |

### 4. Gradual Unfreezing

Instead of unfreezing all layers at once, progressively unfreeze from top to bottom:

1. Train classifier head (base frozen) — 1-2 epochs.
2. Unfreeze the last conv block, train — 2-3 epochs.
3. Unfreeze the second-to-last block, train — 2-3 epochs.
4. Continue until the entire network is unfrozen.

This prevents catastrophic forgetting of early-layer features.

## Domain Adaptation

A special case of transfer learning where the source and target tasks are the same but **data distributions differ**.

### Examples
| Source Domain | Target Domain | Shift |
|---|---|---|
| Real photos | Hand-drawn sketches | Style shift |
| Natural images | Medical X-rays | Content shift |
| Daytime scenes | Nighttime scenes | Illumination shift |

### Approaches
1. **Supervised DA**: Labeled data in both domains. Fine-tune from source to target.
2. **Unsupervised DA**: Labels only in source domain. Use techniques like adversarial training (Gradient Reversal Layer, CycleGAN) to make features domain-invariant.
3. **Self-supervised DA**: Use pretext tasks (rotation prediction, contrastive learning) to learn domain-robust features before fine-tuning.

## Practical Guidelines

| Factor | Feature Extraction | Fine-Tuning |
|---|---|---|
| Target dataset size | < 1K per class | > 1K per class |
| Similarity to pre-training | High | Low or unknown |
| Training time | Fast (minutes-hours) | Slow (hours-days) |
| Overfitting risk | Low | Higher (needs regularization) |
| GPU memory | Lower (no grad through base) | Higher |

### Recommendation Flowchart

```
Is the target dataset similar to ImageNet?
├── YES
│   ├── Dataset small? → Feature extraction
│   └── Dataset large? → Fine-tune base (low LR)
└── NO
    ├── Dataset small? → Feature extraction + strong regularization
    └── Dataset large? → Fine-tune all (standard LR)
```

## Implementation Checklist

1. Choose a pre-trained model (ResNet, EfficientNet, etc.).
2. Remove the original classifier head.
3. Add new layers matching your number of classes.
4. Decide freeze vs. fine-tune strategy.
5. Use a low learning rate for pre-trained weights (1e-5 to 1e-4).
6. Use Adam or SGD with momentum.
7. Monitor validation loss to avoid overfitting.
