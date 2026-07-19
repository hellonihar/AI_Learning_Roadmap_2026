# Exercises: CV using Deep Learning

## Exercise 1: Implement IoU Computation

Write a function `compute_iou(box1, box2)` that computes the Intersection over Union between two bounding boxes in `[x1, y1, x2, y2]` format (top-left and bottom-right corners).

```python
def compute_iou(box1, box2):
    # Your code here
    pass
```

**Test cases:**
- `box1 = [0, 0, 10, 10]`, `box2 = [5, 5, 15, 15]` → IoU should be 0.1429
- `box1 = [0, 0, 10, 10]`, `box2 = [0, 0, 10, 10]` → IoU should be 1.0
- `box1 = [0, 0, 10, 10]`, `box2 = [20, 20, 30, 30]` → IoU should be 0.0

<details>
<summary>Answer</summary>

```python
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0

# Test cases
print(f"{compute_iou([0, 0, 10, 10], [5, 5, 15, 15]):.4f}")  # 0.1429
print(f"{compute_iou([0, 0, 10, 10], [0, 0, 10, 10]):.4f}")  # 1.0000
print(f"{compute_iou([0, 0, 10, 10], [20, 20, 30, 30]):.4f}")  # 0.0000
```
</details>

---

## Exercise 2: Calculate mAP

Given the following predictions and ground truth for a single class, calculate Average Precision (AP). Each prediction has `[confidence, x1, y1, x2, y2]` and ground truth has `[x1, y1, x2, y2]`. Use IoU threshold of 0.5.

```python
ground_truths = [
    [10, 20, 50, 80],   # GT box 1
    [30, 40, 70, 100],  # GT box 2
]

predictions = [
    [0.95, 12, 22, 48, 78],   # pred A
    [0.80, 28, 38, 72, 102],  # pred B
    [0.60, 100, 120, 150, 180], # pred C (false positive)
    [0.40, 10, 18, 52, 82],   # pred D (duplicate of GT 1)
]
```

1. Sort predictions by confidence.
2. Match each prediction to a GT using IoU ≥ 0.5 (each GT can be matched once).
3. Compute precision and recall at each threshold.
4. Interpolate AP (11-point interpolation: average precision at recall levels 0.0, 0.1, ..., 1.0).

<details>
<summary>Answer</summary>

```python
def compute_ap(gt_boxes, preds, iou_thresh=0.5):
    # Sort predictions by confidence descending
    preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)

    matched_gts = set()
    tp = []
    fp = []

    for conf, *pred_box in preds_sorted:
        best_iou = 0
        best_gt_idx = -1
        for i, gt_box in enumerate(gt_boxes):
            if i in matched_gts:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_thresh:
            tp.append(1)
            fp.append(0)
            matched_gts.add(best_gt_idx)
        else:
            tp.append(0)
            fp.append(1)

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        p = max([precisions[j] for j in range(len(precisions)) if recalls[j] >= t], default=0)
        ap += p / 11
    return ap

ap = compute_ap(ground_truths, predictions)
print(f"AP: {ap:.4f}")  # Expected: ~0.8155
```
</details>

---

## Exercise 3: Design a U-Net for RGB Image Segmentation

Complete the U-Net implementation for segmenting 256×256 RGB images into 5 classes. Ensure the bottleneck has 1024 channels, and each decoder step uses skip connections via channel concatenation.

```python
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=5):
        super().__init__()
        # Define encoder, bottleneck, decoder, and output layers
        # Use DoubleConv: Conv2d → BN → ReLU → Conv2d → BN → ReLU
        pass

    def forward(self, x):
        pass
```

<details>
<summary>Answer</summary>

```python
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.enc1 = DoubleConv(n_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)
```
</details>

---

## Exercise 4: Compare FCN vs U-Net

List four key differences between FCN and U-Net for semantic segmentation.

<details>
<summary>Answer</summary>

| Aspect | FCN | U-Net |
|--------|-----|-------|
| Architecture | Encoder-only with skip connections from intermediate layers to upsampled output | Symmetric encoder-decoder with skip connections at every level |
| Skip connections | Added (element-wise sum) before upsampling | Concatenated (channel-wise) before each decoder block |
| Upsampling | Fixed bilinear interpolation (FCN-32s) or learnable transposed convs (FCN-8s) | Learnable transposed convolutions at every decoder level |
| Parameter count | Lower (VGG backbone + small upsampling layers) | Higher (full decoder doubles parameters) |
| Small object performance | Limited by coarse upsampling | Strong — skip connections preserve fine spatial detail |
| Typical use case | General segmentation | Biomedical / tasks requiring precise boundaries |
</details>

---

## Exercise 5: Implement a Data Augmentation Pipeline

Write a PyTorch `Compose` pipeline for training an object detection model on 640×480 images. Include:

- Random horizontal flip (p=0.5)
- Random brightness/contrast adjustment
- Random scaling (0.8× to 1.2×)
- Normalisation to ImageNet stats
- Conversion to tensor

For object detection, augmentations must also transform bounding boxes. Describe how you would handle this.

<details>
<summary>Answer</summary>

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

detection_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# Usage:
# transformed = detection_transform(image=image, bboxes=boxes, labels=labels)
# transformed_image = transformed['image']
# transformed_boxes = transformed['bboxes']
```

**Handling bounding boxes:** Standard `torchvision.transforms` only applies to images. For detection/segmentation, use `albumentations` (which transforms both images and boxes/ masks simultaneously) or manually apply geometric transforms to both images and boxes. `albumentations.A.Compose` with `bbox_params` automatically handles box transformation for `HorizontalFlip`, `RandomScale`, `Rotate`, etc. The `pascal_voc` format means boxes are `[x_min, y_min, x_max, y_max]`.
</details>

---

## Exercise 6: Implement Dice Loss

Implement the Dice loss function for multi-class segmentation. It should accept raw logits `(N, C, H, W)` and integer targets `(N, H, W)`.

```python
def dice_loss(preds, targets, smooth=1.0):
    # Your code here
    pass
```

<details>
<summary>Answer</summary>

```python
def dice_loss(preds, targets, smooth=1.0):
    num_classes = preds.shape[1]
    preds = torch.softmax(preds, dim=1)  # (N, C, H, W)
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes)
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

    intersection = (preds * targets_one_hot).sum(dim=(2, 3))
    pred_sum = preds.sum(dim=(2, 3))
    target_sum = targets_one_hot.sum(dim=(2, 3))

    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    return 1 - dice.mean()
```
</details>

---

## Exercise 7: Progressive Unfreezing Schedule

Given a ResNet-50 with layer groups `conv1`, `layer1`, `layer2`, `layer3`, `layer4`, and a new classifier head, write a PyTorch training loop that implements a 3-phase progressive unfreezing schedule:

- Phase 1 (epochs 1-5): train only the classifier head at lr=1e-3
- Phase 2 (epochs 6-15): unfreeze `layer4` and `layer3`, train with discriminative LRs
- Phase 3 (epochs 16-30): unfreeze everything, reduce all LRs by 10×

<details>
<summary>Answer</summary>

```python
model = models.resnet50(weights='IMAGENET1K_V1')
model.fc = nn.Linear(2048, 10)

# Phase 1: freeze everything except fc
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

for epoch in range(1, 6):
    train_epoch(model, train_loader, optimizer)
    scheduler.step()

# Phase 2: unfreeze layer3 and layer4
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
optimizer = optim.Adam([
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 5e-5},
    {'params': model.fc.parameters(), 'lr': 1e-3},
])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(6, 16):
    train_epoch(model, train_loader, optimizer)
    scheduler.step()

# Phase 3: unfreeze everything
for param in model.parameters():
    param.requires_grad = True
optimizer = optim.Adam([
    {'params': model.conv1.parameters(), 'lr': 1e-6},
    {'params': model.layer1.parameters(), 'lr': 1e-6},
    {'params': model.layer2.parameters(), 'lr': 1e-6},
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-4},
])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

for epoch in range(16, 31):
    train_epoch(model, train_loader, optimizer)
    scheduler.step()
```
</details>

---

## Exercise 8: Contrastive Learning Loss (NT-Xent)

Implement the NT-Xent (Normalised Temperature-scaled Cross Entropy) loss used in SimCLR. Given a batch of `N` images, two augmented views produce `2N` representations `z`. Positive pairs are `(z[0], z[N])`, `(z[1], z[N+1])`, etc. The loss for a positive pair `(i, j)` is:

```
L(i,j) = -log( exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ) )
```

```python
def nt_xent_loss(z, temperature=0.5):
    # z shape: (2*N, D)
    pass
```

<details>
<summary>Answer</summary>

```python
def nt_xent_loss(z, temperature=0.5):
    z = nn.functional.normalize(z, dim=1)
    N = z.shape[0] // 2

    # Similarity matrix
    sim = torch.matmul(z, z.T) / temperature  # (2N, 2N)

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * N, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    # Positive pairs: (0, N), (1, N+1), ..., (N-1, 2N-1) and reverse
    pos_sim = torch.cat([
        torch.diag(sim, N),   # i, i+N
        torch.diag(sim, -N),  # i+N, i
    ])  # (2N,)

    # Loss
    loss = -pos_sim + torch.logsumexp(sim, dim=1)
    return loss.mean()
```
</details>
