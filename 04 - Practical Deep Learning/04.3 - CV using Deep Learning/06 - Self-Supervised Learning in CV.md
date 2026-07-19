# Self-Supervised Learning in CV

## Why Self-Supervised Learning?

Labelled data is expensive. A radiologist labelling medical images costs $100+/hour; annotating a single Cityscapes segmentation mask takes 90 minutes. **Self-supervised learning (SSL)** eliminates the need for human annotations by creating supervisory signals from the data itself. The model first learns rich visual representations through a pretext task on unlabelled data, then transfers these representations to downstream tasks with minimal labelled examples.

## Contrastive Learning

Contrastive methods pull representations of similar images (positive pairs) closer and push representations of dissimilar images (negative pairs) apart.

### SimCLR

SimCLR (Chen et al., 2020) is a simple and influential framework:

1. **Augmentation** — apply two random augmentations (random crop, colour jitter, Gaussian blur, horizontal flip) to each image, producing a positive pair.
2. **Encoder** — a ResNet-50 processes both augmented views into representation vectors.
3. **Projection head** — a small MLP maps representations to a latent space where contrastive loss is applied.
4. **NT-Xent loss** (Normalised Temperature-scaled Cross Entropy) — for each positive pair in a batch of `N` images, treat the other `2(N-1)` augmented views as negatives:

```
L_i = -log( exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ) )
```

where `sim()` is cosine similarity and `τ` is a temperature parameter.

**Key insight:** large batch size (4,096) and strong augmentations are critical. SimCLR matches supervised ImageNet performance with just 1% of labels.

### MoCo (Momentum Contrast)

MoCo (He et al., 2020) addresses the batch size limitation by maintaining a **queue** of negative samples across batches and a **momentum encoder** that slowly tracks the main encoder's weights:

```python
@torch.no_grad()
def momentum_update(main_encoder, momentum_encoder, m=0.999):
    for param_main, param_momentum in zip(main_encoder.parameters(), momentum_encoder.parameters()):
        param_momentum.data = param_momentum.data * m + param_main.data * (1 - m)
```

This decouples batch size from the number of negatives and enables training with smaller batches (256 vs 4,096).

## Masked Image Modelling (MAE)

**Masked Autoencoders** (He et al., 2022) adapt the BERT approach to vision. The model:

1. Divides an image into patches (e.g., 16×16).
2. Randomly masks 75% of patches.
3. Encodes only the visible patches with a ViT encoder.
4. Reconstructs the original pixels of the masked patches with a lightweight decoder.

MAE is surprisingly simple but highly effective. The high masking ratio (75%) forces the model to learn genuine semantic understanding rather than just interpolating nearby pixels. Fine-tuned MAE achieves 87.8% top-1 accuracy on ImageNet with ViT-Huge.

## Pretext Tasks

Pretext tasks defined SSL before contrastive learning became dominant:

| Task | Description | Downstream Benefit |
|------|-------------|-------------------|
| **Rotation prediction** | Predict the rotation angle (0°, 90°, 180°, 270°) applied to the image | Learns object orientation and canonical features |
| **Jigsaw puzzle** | Shuffle image patches; predict the original arrangement | Learns spatial relationships and part-whole hierarchies |
| **Context prediction** | Predict the relative position of two patches | Learns spatial context |
| **Colourisation** | Predict colour channels from grayscale input | Learns object shape and texture cues |
| **Relative patch location** | Predict which of 8 neighbours a patch came from | Learns local spatial structure |

These tasks are less performant than contrastive or MAE approaches but are conceptually important and computationally cheaper.

## Practical Workflow

```
Phase 1: Pre-train on unlabelled data (SSL)
  [100K unlabelled medical X-rays] → [SimCLR/MoCo/MAE] → [Encoder weights]

Phase 2: Linear probing (evaluation protocol)
  [Encoder (frozen)] → [Linear classifier] → [Train on 100 labelled samples]
  High accuracy = good representations

Phase 3: Fine-tune on downstream task
  [Encoder (unfrozen)] → [Task-specific head] → [Train on 1K labelled samples]
```

Linear probing (freezing the encoder and training only a linear classifier) is the standard evaluation protocol. Strong linear probing performance on ImageNet indicates generalisable representations.

## Why SSL Reduces Annotation Cost

| Scenario | Labelled Samples | Accuracy |
|----------|-----------------|----------|
| From scratch | 1,000 | ~50% |
| Fine-tuned SimCLR | 1,000 | ~85% |
| Fine-tuned SimCLR | 10,000 | ~92% |
| From scratch | 50,000 | ~90% |

SSL pre-training on unlabelled data effectively reduces the labelled data requirement by **10-100×**. In domains like medical imaging or remote sensing where labels are scarce, SSL is transformative.

## Current State (2026)

- **DINOv2** (Meta, 2023) produces features good enough for k-NN classifiers without fine-tuning.
- **ImageBind** learns joint embeddings across 6 modalities (images, text, audio, depth, thermal, IMU).
- SSL features now outperform supervised pre-training on most transfer learning benchmarks.

## Summary

Self-supervised learning eliminates the annotation bottleneck. Contrastive methods (SimCLR, MoCo) learn by pulling augmented views of the same image together. Masked image modelling (MAE) learns by reconstructing masked patches. Both produce representations that reduce labelled data requirements by orders of magnitude, making SSL a critical tool for any practical CV practitioner.
