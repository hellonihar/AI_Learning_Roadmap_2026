# Code Walkthrough: Transfer Learning with ResNet18 on CIFAR-10

This walkthrough implements feature extraction and fine-tuning variants of ResNet-18 on CIFAR-10, with training loops, evaluation, and Grad-CAM visualisation.

## 1. Setup & Data Preparation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
EPOCHS_FE = 20   # feature extraction
EPOCHS_FT = 30   # fine-tuning
```

## 2. Augmentation Pipeline

CIFAR-10 images are 32×32, but ImageNet models expect 224×224. We resize and apply augmentation:

```python
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
```

Note: ImageNet normalisation statistics differ from CIFAR-10's own stats. Because we use pre-trained weights, we must use ImageNet's statistics.

## 3. Model Setup with ResNet-18

```python
def get_model(mode='feature_extraction'):
    model = models.resnet18(weights='IMAGENET1K_V1')

    if mode == 'feature_extraction':
        for param in model.parameters():
            param.requires_grad = False

    # Replace classifier head
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 10),
    )

    return model
```

## 4. Training Loop

```python
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc
```

## 5. Feature Extraction Training

```python
model = get_model('feature_extraction').to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(EPOCHS_FE):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, test_loader, criterion)
    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f'Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')
```

Expected: ~82–85% validation accuracy after 20 epochs, plateauing quickly since only the head is trained.

## 6. Fine-Tuning (Phase 2)

```python
# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# Lower learning rate for the backbone
optimizer = optim.Adam([
    {'params': [p for n, p in model.named_parameters() if 'fc' not in n], 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-4},
])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FT)

for epoch in range(EPOCHS_FT):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, test_loader, criterion)
    scheduler.step()
    print(f'Fine-tune Epoch {epoch+1:02d} | Val Acc: {val_acc:.2f}%')
```

Expected: ~90–93% validation accuracy after fine-tuning, significantly higher than feature extraction alone.

## 7. Accuracy/Loss Curves

```python
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses + [np.nan]*EPOCHS_FT, label='Train Loss')
plt.plot(val_losses + [np.nan]*EPOCHS_FT, label='Val Loss')
plt.axvline(x=EPOCHS_FE-1, color='gray', linestyle='--', label='Start Fine-tune')
plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accs + [np.nan]*EPOCHS_FT, label='Train Acc')
plt.plot(val_accs + [np.nan]*EPOCHS_FT, label='Val Acc')
plt.axvline(x=EPOCHS_FE-1, color='gray', linestyle='--', label='Start Fine-tune')
plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
plt.show()
```

The accuracy curve typically shows a sharp initial rise during feature extraction, a plateau, then a second rise after unfreezing.

## 8. Grad-CAM Visualisation (Conceptual)

Grad-CAM produces a heatmap showing which regions of the input image influenced the model's prediction most strongly.

### Algorithm

1. Forward pass the image and obtain the logits for the target class.
2. Zero all other gradients. Backward from the target class logit to the last convolutional layer's feature map.
3. Compute **gradient weights**: `α_c = 1/Z × Σᵢ Σⱼ ∂y^c / ∂Aᵢⱼ` (global average pooling of gradients).
4. Compute **weighted combination**: `L_c = ReLU(Σ α_c · A)`.
5. Upsample `L_c` to the input image size and overlay.

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        # Register hooks
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x, class_idx=None):
        # Forward pass
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        # Weighted combination
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Normalize and resize
        cam = cam.squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=x.shape[2:], mode='bilinear', align_corners=False
        )
        return cam.squeeze()
```

### Usage

```python
target_layer = model.layer4[-1]  # last conv layer in ResNet18
gradcam = GradCAM(model, target_layer)

image, label = test_set[0]
image_tensor = image.unsqueeze(0).to(device)
cam = gradcam.generate(image_tensor)

# Overlay heatmap
heatmap = cam.cpu().detach().numpy()
plt.imshow(image.permute(1, 2, 0).cpu().numpy())
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.title(f'Predicted: {classes[image_tensor.argmax()]}')
plt.show()
```

Grad-CAM reveals whether the model is looking at the correct object or relying on spurious correlations (e.g., background cues), making it an essential diagnostic tool.

## Summary

This walkthrough demonstrated the complete transfer learning workflow: data augmentation, ResNet-18 feature extraction (~85% accuracy), fine-tuning (~92% accuracy), training visualisation, and Grad-CAM interpretability. The gap between feature extraction and fine-tuning accuracy indicates how much the dataset benefits from adapting pre-trained features.
