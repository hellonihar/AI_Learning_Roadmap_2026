# Object Detection

## Problem Formulation

Object detection combines classification with localisation. For each image, the model outputs a variable number of detections, each containing:

- **Bounding box** — `(x₁, y₁, x₂, y₂)` coordinates (top-left and bottom-right corners) or `(cx, cy, w, h)` (centre + dimensions).
- **Class label** — a probability distribution over `C` object classes plus a background class.
- **Confidence score** — how likely the box contains an object.

This is fundamentally harder than classification because the model must handle overlapping objects, different scales, and varying numbers of objects per image.

## Key Concepts

### Intersection over Union (IoU)

IoU measures the overlap between a predicted box and a ground-truth box:

```
IoU = Area of Overlap / Area of Union
```

A detection is considered a **true positive** if IoU ≥ threshold (typically 0.5) and the class label matches. IoU is used in both training (matching predictions to ground truth) and evaluation.

```python
def compute_iou(box1, box2):
    """box = [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0
```

### Mean Average Precision (mAP)

mAP is the standard evaluation metric for object detection:

1. Compute **precision** and **recall** for each class by sorting all detections by confidence score.
2. Generate a precision-recall curve and compute **Average Precision (AP)** — the area under this curve.
3. **mAP** = mean of AP across all classes.

Common variants: `mAP@0.5` (IoU ≥ 0.5) and `mAP@0.5:0.95` (average over IoU thresholds from 0.5 to 0.95 in steps of 0.05 — the COCO standard).

## Detection Architectures

### R-CNN Family

| Model | Key Idea | Speed |
|-------|----------|-------|
| **R-CNN** | Selective search → crop → classify each region | Very slow |
| **Fast R-CNN** | Single forward pass + RoI pooling | Faster |
| **Faster R-CNN** | Region Proposal Network (RPN) learns to propose boxes | End-to-end, ~7 FPS |

**Faster R-CNN** remains widely used. It consists of:
- A **backbone** CNN (e.g., ResNet-50) producing a feature map.
- A **Region Proposal Network (RPN)** that slides a small network over the feature map to predict object boundaries and objectness scores at each location.
- **RoI Align** crops features for each proposal and resizes them to a fixed size.
- A **two-branch head** that predicts class probabilities and refines box coordinates.

```python
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
model.eval()

# Inference
image_tensor = preprocess(image)  # torchvision.transforms.ToTensor() + Normalize
with torch.no_grad():
    predictions = model([image_tensor])

# predictions[0] contains:
#   'boxes': Tensor[N, 4]
#   'labels': Tensor[N]
#   'scores': Tensor[N]

# Filter by confidence
keep = predictions[0]['scores'] > 0.5
boxes = predictions[0]['boxes'][keep]
labels = predictions[0]['labels'][keep]
```

### YOLO (You Only Look Once)

YOLO reframes detection as a **single regression problem** — one forward pass predicts all boxes and class probabilities simultaneously.

| Version | Key Innovation |
|---------|---------------|
| YOLOv1 | Grid-based prediction (7×7 grid, 2 boxes per cell) |
| YOLOv3 | Multi-scale predictions, darknet backbone, 3 box sizes per cell |
| YOLOv5 | PyTorch-native, mosaic augmentation, auto-anchor optimisation |
| YOLOv8 | Anchor-free, decoupled head, task-aligned assigner |

YOLO divides the image into an `S×S` grid. Each grid cell predicts `B` bounding boxes, each with confidence and class probabilities. The loss combines box regression (CIoU), objectness, and classification terms. YOLOv8 reaches **~45 mAP@0.5:0.95 at 150+ FPS** on COCO.

### SSD (Single Shot MultiBox Detector)

SSD predicts boxes from multiple feature maps at different scales, enabling detection of both small and large objects without a separate proposal network. It is simpler than Faster R-CNN but less accurate than modern YOLO variants.

## Comparison

| Criteria | Faster R-CNN | YOLOv8 | SSD |
|----------|-------------|--------|-----|
| Type | Two-stage | One-stage | One-stage |
| Speed | Moderate (7-15 FPS) | Very fast (150+ FPS) | Fast (60+ FPS) |
| Accuracy | High (47 mAP) | Higher (45 mAP) | Moderate (25 mAP) |
| Small objects | Good (RPN helps) | Good (multi-scale) | Weak |
| Ease of use | torchvision built-in | Ultralytics package | Deprecated in torchvision |

## Practical Considerations

1. **Anchor boxes** — pre-defined bounding box shapes at each location. Most frameworks auto-compute optimal anchors from your dataset using k-means on ground-truth boxes.
2. **NMS (Non-Maximum Suppression)** — removes duplicate detections by keeping the highest-confidence box and suppressing others with IoU > threshold (typically 0.5).
3. **Data augmentation** — mosaic, mixup, random affine, and HSV jittering improve detection robustness significantly.
4. **Loss functions** — combine classification loss (cross-entropy) with regression loss (L1, smooth L1, or CIoU for boxes).

## Summary

Object detection localises and classifies multiple objects. IoU and mAP are the fundamental evaluation tools. Two-stage detectors (Faster R-CNN) offer accuracy and are easy to use with torchvision; one-stage detectors (YOLO) trade marginal accuracy for massive speed gains. For most applications, start with YOLOv8 or torchvision's Faster R-CNN with a ResNet-50 backbone.
