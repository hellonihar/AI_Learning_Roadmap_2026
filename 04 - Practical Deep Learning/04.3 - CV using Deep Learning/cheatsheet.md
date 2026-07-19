# CV using Deep Learning — Cheatsheet

## Classification Pipeline

```
Data → Augmentation → DataLoader → Model (CNN) → Loss (CrossEntropy) → Backprop → Optimizer (Adam/SGD)
```

| Step | Key Code |
|------|----------|
| Transforms | `transforms.Compose([Resize, RandomHorizontalFlip, ToTensor, Normalize])` |
| Loader | `DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)` |
| Model | `nn.Conv2d(in, out, 3, padding=1)` → `BN` → `ReLU` → `MaxPool2d(2)` |
| Classifier | `AdaptiveAvgPool2d(1)` → `Flatten` → `Dropout` → `Linear` |
| Loss | `nn.CrossEntropyLoss()` |
| Optimizer | `optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)` |
| Scheduler | `CosineAnnealingLR(optimizer, T_max=200)` |

### Regularisation Checklist
- [ ] RandomHorizontalFlip / RandomCrop / ColorJitter
- [ ] Weight decay (Adam: 5e-4, SGD: 1e-4)
- [ ] Dropout (0.3–0.5) before linear layers
- [ ] BatchNorm (reduces internal covariate shift)
- [ ] Label smoothing (`nn.CrossEntropyLoss(label_smoothing=0.1)`)

### CIFAR-10 Normalisation
```python
Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
```

### ImageNet Normalisation (for pre-trained models)
```python
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

---

## Transfer Learning Strategies

| Strategy | Frozen Layers | Trainable | LR | When to Use |
|----------|--------------|-----------|-----|-------------|
| Feature extraction | All conv layers | Classifier head only | 1e-3 (head) | Small dataset, similar domain |
| Fine-tuning (full) | None | All | 1e-4 (backbone), 1e-3 (head) | Large dataset, domain shift |
| Progressive unfreezing | Gradual unfreeze | Increasing | Discriminative | Medium dataset, careful training |

### Discriminative LR Setup
```python
optimizer = optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-6},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3},
])
```

### torchvision Models
```python
models.resnet18(weights='IMAGENET1K_V1')
models.resnet50(weights='IMAGENET1K_V1')
models.efficientnet_b0(weights='IMAGENET1K_V1')
models.efficientnet_b3(weights='IMAGENET1K_V1')
models.convnext_tiny(weights='IMAGENET1K_V1')
```

---

## Object Detection

### Metrics
- **IoU**: `intersection / union` of two boxes
- **mAP@0.5**: mean AP at IoU threshold 0.5
- **mAP@0.5:0.95**: mean AP averaged over IoU thresholds 0.50–0.95 (COCO standard)

### Architecture Quick Reference

| Model | Type | Backbone | FPS | mAP@0.5:0.95 |
|-------|------|----------|-----|---------------|
| Faster R-CNN | Two-stage | ResNet-50-FPN | ~7 | 37 |
| YOLOv8n | One-stage | CSPDarknet | ~150 | 37.3 |
| YOLOv8x | One-stage | CSPDarknet | ~40 | 53.9 |
| SSD300 | One-stage | VGG-16 | ~60 | 25 |

### Inference with torchvision
```python
model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
model.eval()
with torch.no_grad():
    preds = model([image_tensor])
keep = preds[0]['scores'] > 0.5
boxes = preds[0]['boxes'][keep]
labels = preds[0]['labels'][keep]
```

### NMS
```python
from torchvision.ops import nms
keep = nms(boxes, scores, iou_threshold=0.5)
```

---

## Semantic Segmentation

### Architectures

| Architecture | Key Feature | Best For |
|-------------|-------------|----------|
| FCN | Replace FC with 1×1 convs + upsampling | Understanding the concept |
| U-Net | Encoder-decoder with skip connections | Small datasets, precise boundaries |
| DeepLabv3+ | Atrous convs + ASPP + encoder-decoder | Multi-scale context, large datasets |

### Loss Functions
```python
# Standard
nn.CrossEntropyLoss()

# Combined (recommended)
ce_loss = nn.CrossEntropyLoss()
dice = dice_loss(preds, targets)
total_loss = ce_loss(preds, targets) + dice
```

### Evaluation
```python
# Per-class IoU → mIoU
IoU_c = TP_c / (TP_c + FP_c + FN_c)
mIoU = mean(IoU_c for c in valid_classes)
```

---

## Data Augmentation

### torchvision Transforms
```python
transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

### Albumentations (for detection/segmentation)
```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomScale(scale_limit=0.2),
    A.ColorJitter(brightness=0.2, contrast=0.2),
    A.Normalize(mean, std),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
```

---

## Common Training Settings

| Parameter | Classification | Detection | Segmentation |
|-----------|---------------|-----------|--------------|
| Batch size | 128–256 | 16–64 | 8–32 |
| LR (Adam) | 1e-3 | 1e-4 | 1e-4 |
| LR fine-tuning | 1e-4 | 1e-5 | 1e-5 |
| Weight decay | 5e-4 | 1e-4 | 1e-4 |
| Scheduler | CosineAnnealing | StepLR (×0.1 @ 60%) | CosineAnnealing |
| Epochs | 100–300 | 100–300 | 100–500 |

---

## Evaluation Metrics Summary

| Task | Primary Metric | Secondary Metrics |
|------|---------------|-------------------|
| Classification | Top-1 Accuracy | Top-5, Precision, Recall, F1, Confusion Matrix |
| Object Detection | mAP@0.5:0.95 | mAP@0.5, AP per class, AR (Average Recall) |
| Segmentation | mIoU | Dice coefficient, Pixel Accuracy, Boundary F1 |
| Generation | FID (Fréchet Inception Distance) | IS (Inception Score) |
