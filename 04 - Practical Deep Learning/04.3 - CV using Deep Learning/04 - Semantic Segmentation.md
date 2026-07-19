# Semantic Segmentation

## Problem Definition

Semantic segmentation assigns a class label to **every pixel** in an image. The output is a segmentation map of shape `(H, W)` where each pixel value is the class index, or `(C, H, W)` where each channel represents the probability for one class. Unlike object detection, segmentation does not distinguish between object instances — all pixels belonging to "car" are labelled identically regardless of which car they belong to.

## Key Architectures

### Fully Convolutional Network (FCN)

The seminal work that replaced fully-connected layers with convolutional layers, enabling arbitrary-sized inputs. An FCN:

1. Encodes the image through a standard CNN backbone (e.g., VGG-16) but replaces the classifier with 1×1 convolutions.
2. Upsamples the coarse feature map back to the original resolution using **transposed convolutions** (learned upsampling) or bilinear interpolation.
3. Uses **skip connections** to combine coarse semantic information from deep layers with fine spatial detail from shallow layers (FCN-8s, FCN-16s, FCN-32s variants).

**Limitation:** predictions can be coarse due to fixed upsampling.

### U-Net

U-Net extends FCN with a symmetric **encoder-decoder** architecture connected by **skip connections**, designed originally for biomedical segmentation.

```
Encoder (downsampling)         Decoder (upsampling)
Input: 3×256×256
    │                                │
[Conv→BN→ReLU]×2                [UpConv→BN→ReLU]×2
    │ MaxPool (skip─►concat)         │
[Conv→BN→ReLU]×2                [UpConv→BN→ReLU]×2
    │ MaxPool (skip─►concat)         │
[Conv→BN→ReLU]×2                [UpConv→BN→ReLU]×2
    │ MaxPool (skip─►concat)         │
[Conv→BN→ReLU]×2         Bottleneck  │
    │ MaxPool (skip─►concat)         │
[Conv→BN→ReLU]×2                [UpConv→BN→ReLU]×2
    │                             1×1 Conv → C classes
```

Each decoder step concatenates features from the corresponding encoder layer via skip connections, preserving fine spatial details lost during downsampling. This is critical for precise boundary delineation.

```python
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(n_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
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

### DeepLab Family

| Variant | Key Innovation |
|---------|---------------|
| DeepLabv1 | Atrous (dilated) convolutions for larger receptive field without downsampling |
| DeepLabv2 | **ASPP** (Atrous Spatial Pyramid Pooling) — parallel atrous convs at different rates |
| DeepLabv3 | Improved ASPP + image-level features |
| DeepLabv3+ | Encoder-decoder with depthwise separable convs |
| **Key Advantage** | Captures multi-scale context without resolution loss |

Atrous convolutions insert "holes" between kernel elements to enlarge the receptive field at the same computational cost. ASPP runs multiple atrous convolutions in parallel (rates 6, 12, 18, 24) and concatenates the results.

## Loss Functions

### Pixel-wise Cross-Entropy

The standard choice. Each pixel contributes equally to the loss:

```python
criterion = nn.CrossEntropyLoss()  # applies softmax internally
loss = criterion(predictions, targets)  # predictions: (N, C, H, W), targets: (N, H, W)
```

Fails when classes are heavily imbalanced (e.g., 95% background, 5% object).

### Dice Loss

Based on the Dice coefficient (similar to IoU but differentiable):

```python
def dice_loss(preds, targets, smooth=1.0):
    preds = torch.softmax(preds, dim=1)
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=preds.shape[1])
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

    intersection = (preds * targets_one_hot).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()
```

**Combined loss** (`CrossEntropy + Dice`) is the most common in practice, balancing pixel accuracy with region-level overlap.

## Evaluation: IoU (Jaccard Index)

Per-class IoU is the standard metric, computed on the discretised predictions:

```
IoU_c = TP_c / (TP_c + FP_c + FN_c)
```

```python
def compute_iou_per_class(preds, targets, num_classes):
    ious = []
    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        iou = intersection / union if union > 0 else float('nan')
        ious.append(iou)
    return ious  # mean of these = mIoU
```

**Mean IoU (mIoU)** is the aggregate metric (average over classes). On common benchmarks:
- **PASCAL VOC**: ~85% mIoU (state-of-the-art)
- **Cityscapes** (urban driving, 19 classes): ~85% mIoU

## Summary

Semantic segmentation performs pixel-level classification. U-Net with its encoder-decoder + skip connection design is the go-to architecture for tasks with limited data. DeepLab excels at multi-scale context. Combined CrossEntropy + Dice loss is recommended for imbalanced classes. mIoU is the standard evaluation metric.
