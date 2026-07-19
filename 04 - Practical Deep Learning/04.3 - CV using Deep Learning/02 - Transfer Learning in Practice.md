# Transfer Learning in Practice

## Why Transfer Learning?

Training CNNs from scratch requires massive labelled datasets (ImageNet has 1.2M images) and days of GPU compute. Transfer learning reuses features learned on ImageNet for a new task, dramatically reducing data requirements and training time. It is the default approach for almost all real-world CV projects.

## torchvision Models

`torchvision.models` provides pre-trained weights for dozens of architectures:

```python
from torchvision import models

# Load with pre-trained ImageNet weights
resnet18 = models.resnet18(weights='IMAGENET1K_V1')
efficientnet_b0 = models.efficientnet_b0(weights='IMAGENET1K_V1')
```

Popular choices:

| Model | Params | Top-1 | Speed | Use Case |
|-------|--------|-------|-------|----------|
| ResNet-18 | 11M | 69.8% | Fast | General purpose, limited compute |
| ResNet-50 | 25M | 76.1% | Medium | Balanced accuracy/speed |
| EfficientNet-B0 | 5.3M | 77.7% | Fast | Best accuracy-per-parameter |
| EfficientNet-B3 | 12M | 81.1% | Medium | High accuracy |
| ConvNeXt-T | 28M | 82.1% | Medium | Modern architecture |

## Feature Extraction vs Fine-Tuning

Two main strategies exist, and the choice depends on dataset size and similarity to ImageNet.

### Feature Extraction

Freeze the convolutional backbone and train only the newly added classifier head:

```python
model = models.resnet18(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully-connected layer for 10 classes
model.fc = nn.Linear(model.fc.in_features, 10)

# Only fc parameters are trained
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
```

**When to use:** Small dataset (< 1,000 images per class), or when the new task is visually similar to ImageNet (natural images, objects).

### Fine-Tuning

All layers are updated with a lower learning rate, allowing the model to adapt features to the new domain:

```python
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 10)

# All parameters trainable
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 10x lower than from-scratch
```

**When to use:** Large dataset (> 5,000 images per class), or domain shift (medical images, satellite imagery, thermal images).

## Freezing Layers and Progressive Unfreezing

### Layer Freezing

Inspect and selectively freeze:

```python
# Freeze first two layer groups
ct = 0
for name, child in model.named_children():
    ct += 1
    if ct < 3:  # freeze first 2 blocks
        for param in child.parameters():
            param.requires_grad = False
```

### Progressive Unfreezing

A three-phase strategy that stabilises training:

```python
# Phase 1: train only the new classifier head (5 epochs)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 10)
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

# Phase 2: unfreeze last block, continue training (10 epochs)
for param in model.layer4.parameters():
    param.requires_grad = True
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3},
])

# Phase 3: unfreeze everything with lower lr (10 epochs)
for param in model.parameters():
    param.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=1e-5)
```

## Discriminative Learning Rates

Different layers learn different features. Early layers detect generic edges and colours; later layers learn task-specific patterns. **Discriminative learning rates** assign lower LR to early layers and higher LR to later layers:

```python
optimizer = optim.Adam([
    {'params': model.conv1.parameters(), 'lr': 1e-6},
    {'params': model.layer1.parameters(), 'lr': 1e-6},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3},
])
```

This is the technique used in the ULMFiT paper and is well-supported by `fastai` and PyTorch.

## Practical Tips

1. **Use the same preprocessing** as the pre-trained model. `torchvision.models` expects `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
2. **Resize inputs** to at least 224×224 — the native resolution of ImageNet pre-trained models. Use `transforms.Resize(256)` then `transforms.CenterCrop(224)`.
3. **Start with feature extraction.** Add fine-tuning only if validation accuracy plateaus. This avoids destroying useful pre-trained features.
4. **Lower learning rates** are critical for fine-tuning. A rate of `1e-4` to `1e-5` is typical vs `1e-3` for from-scratch.
5. **Monitor feature drift** — if early-layer weights change dramatically during fine-tuning, your dataset may be too different from ImageNet. Consider a different pre-training source (e.g., medical models for medical data).

## When Transfer Learning Fails

- **Domain mismatch:** Pre-trained on natural images, applied to medical X-rays or document scans. Consider self-supervised pre-training on in-domain data.
- **Dataset too small for fine-tuning:** Stick to feature extraction with strong regularisation.
- **New classes are fine-grained (bird species, car models):** Fine-tuning with discriminative LRs works well here, but ensure enough samples per class.

## Summary

Transfer learning is the single most practical technique in applied CV. Feature extraction is fast and works for small datasets; fine-tuning adapts features for larger or domain-shifted datasets. Progressive unfreezing and discriminative learning rates give fine-grained control over the training process. Start with these strategies on every new vision project.
