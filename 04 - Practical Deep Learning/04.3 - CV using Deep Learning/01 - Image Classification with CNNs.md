# Image Classification with CNNs

## The Problem

Image classification assigns a single label to an entire image. Given an input image of shape `(3, H, W)`, a CNN outputs a probability distribution over `K` classes. The CIFAR-10 dataset is our running example: 60,000 `32×32` colour images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), split into 50,000 training and 10,000 test images.

## CNN Architecture Design

A standard CNN stacks three kinds of operations:

- **Convolutional layers** — learn spatial filters that detect edges, textures, and higher-level patterns.
- **Pooling layers** — downsample spatial dimensions (max-pooling `2×2` is common) to build translation invariance and reduce computation.
- **Fully-connected (linear) layers** — act as a classifier on top of the learned feature maps.

### A CIFAR-10 Architecture

```
Conv2d(3, 32, 3, padding=1) → BatchNorm2d → ReLU → MaxPool2d(2)
Conv2d(32, 64, 3, padding=1) → BatchNorm2d → ReLU → MaxPool2d(2)
Conv2d(64, 128, 3, padding=1) → BatchNorm2d → ReLU → MaxPool2d(2)
AdaptiveAvgPool2d(1) → Flatten → Dropout(0.5) → Linear(128, 10)
```

Key design choices:
- **Small kernels (`3×3`)** — stack multiple small kernels to get large receptive fields with fewer parameters. Two `3×3` layers have the same receptive field as one `5×5` but with 28% fewer parameters.
- **BatchNorm after every conv** — stabilises training and allows higher learning rates. It normalises layer outputs to zero mean and unit variance, reducing internal covariate shift.
- **Progressive channel doubling (32→64→128)** — deeper layers learn more abstract features and benefit from higher capacity.
- **Global average pooling** replaces large linear layers, reducing overfitting by eliminating millions of parameters from the classifier.
- **Dropout (0.5)** before the final classifier adds regularisation by randomly deactivating half the neurons each forward pass.

### Why This Architecture Works for CIFAR-10

The 32×32 input is small, so three pooling layers reduce spatial dimensions to 4×4 before global pooling. The 128 channels in the deepest layer provide sufficient representational capacity for 10 classes. Total parameter count is approximately 150K — far smaller than a fully-connected network that would require millions of parameters for the same input size.

### Initialisation

Weight initialisation matters. Use Kaiming He initialisation for ReLU-based networks:

```python
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(init_weights)
```

PyTorch's default initialisation is already reasonable, but explicit control helps reproducibility.

## Training Pipeline

### Data Loading with `ImageFolder` and `DataLoader`

PyTorch's `torchvision.datasets.CIFAR10` inherits from `Dataset`. For custom data, organise folders as:

```
data/
├── train/
│   ├── cat/     (all cat images)
│   ├── dog/     (all dog images)
│   └── ...
└── val/
    ├── cat/
    ├── dog/
    └── ...
```

Then use `ImageFolder`:

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
```

### Transforms for Overfitting Prevention

| Transform | Purpose |
|-----------|---------|
| `RandomHorizontalFlip` | Data augmentation — doubles training variety |
| `RandomCrop` with padding | Translational invariance |
| `ColorJitter` | Invariance to lighting changes |
| `Normalize` | Centre inputs at zero, unit variance — helps optimisation |

CIFAR-10 mean/std values are pre-computed from the entire training set.

### Training Loop

```python
model = CIFAR10CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(200):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

### Overfitting Prevention Checklist

1. **Data augmentation** — the single most effective regulariser for images.
2. **Weight decay (L2 regularisation)** — penalises large weights; `5e-4` is a good starting point.
3. **Dropout** — randomly drops neurons during training.
4. **Early stopping** — monitor validation loss and stop when it plateaus.
5. **Reduce model capacity** — fewer channels or layers if overfitting persists.
6. **Label smoothing** — replaces hard 0/1 targets with soft values (e.g., 0.1/0.9) to prevent overconfidence.

## Evaluation

```python
model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')
```

A well-tuned CNN of this design reaches **~85–88%** on CIFAR-10. For comparison, a plain feed-forward network on raw pixels barely hits 40%, demonstrating the power of convolutional inductive biases.

### Learning Rate Scheduling

The learning rate schedule significantly impacts final accuracy:

- **Cosine annealing** smoothly reduces LR from the initial value to near-zero following a cosine curve. This allows the model to converge to a better minimum than step decay.
- **One-cycle policy** (Leslie Smith) rapidly increases LR then decreases it, providing both fast initial progress and fine-grained convergence.
- **Warmup** — linearly increase LR from 0 to the target over the first 5-10 epochs. This prevents early training instability, especially with BatchNorm layers.

```python
# Cosine annealing with warmup
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Or use warmup manually
for epoch in range(warmup_epochs):
    lr = base_lr * (epoch + 1) / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

## Confusion Matrix & Visualisation

Beyond top-1 accuracy, examine per-class performance:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
```

Misclassifications often occur between semantically similar pairs (cat↔dog, deer↔horse). This informs whether more data or architectural changes are needed for specific classes.

## Summary

A CNN for image classification requires careful co-design of architecture, data pipeline, and regularisation. The inductive biases of convolutions — locality, weight sharing, and translation equivariance — make them far more sample-efficient than fully-connected networks for vision tasks. Mastering this pipeline is the foundation for all advanced CV topics that follow.
