# Resources: CV using Deep Learning

## Foundational Papers

### Image Classification
- **AlexNet** (Krizhevsky et al., 2012) — *ImageNet Classification with Deep Convolutional Neural Networks* — [Link](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- **VGGNet** (Simonyan & Zisserman, 2015) — *Very Deep Convolutional Networks for Large-Scale Image Recognition* — [Link](https://arxiv.org/abs/1409.1556)
- **ResNet** (He et al., 2016) — *Deep Residual Learning for Image Recognition* — [Link](https://arxiv.org/abs/1512.03385)
- **EfficientNet** (Tan & Le, 2019) — *Rethinking Model Scaling for Convolutional Neural Networks* — [Link](https://arxiv.org/abs/1905.11946)

### Transfer Learning
- **ULMFiT** (Howard & Ruder, 2018) — *Universal Language Model Fine-tuning for Text Classification* — introduced discriminative LRs and progressive unfreezing — [Link](https://arxiv.org/abs/1801.06146)
- **TARL** — *Task Adaptive Representation Learning* (used in fastai)

### Object Detection
- **R-CNN** (Girshick et al., 2014) — *Rich feature hierarchies for accurate object detection and semantic segmentation* — [Link](https://arxiv.org/abs/1311.2524)
- **Fast R-CNN** (Girshick, 2015) — [Link](https://arxiv.org/abs/1504.08083)
- **Faster R-CNN** (Ren et al., 2016) — *Towards Real-Time Object Detection with Region Proposal Networks* — [Link](https://arxiv.org/abs/1506.01497)
- **YOLOv1** (Redmon et al., 2016) — *You Only Look Once: Unified, Real-Time Object Detection* — [Link](https://arxiv.org/abs/1506.02640)
- **YOLOv3** (Redmon & Farhadi, 2018) — [Link](https://arxiv.org/abs/1804.02767)
- **YOLOv8** — Ultralytics — [Link](https://github.com/ultralytics/ultralytics)
- **SSD** (Liu et al., 2016) — *SSD: Single Shot MultiBox Detector* — [Link](https://arxiv.org/abs/1512.02325)

### Semantic Segmentation
- **FCN** (Long et al., 2015) — *Fully Convolutional Networks for Semantic Segmentation* — [Link](https://arxiv.org/abs/1411.4038)
- **U-Net** (Ronneberger et al., 2015) — *U-Net: Convolutional Networks for Biomedical Image Segmentation* — [Link](https://arxiv.org/abs/1505.04597)
- **DeepLab** (Chen et al., 2017) — *DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs* — [Link](https://arxiv.org/abs/1606.00915)
- **DeepLabv3+** (Chen et al., 2018) — *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation* — [Link](https://arxiv.org/abs/1802.02611)

### Image Generation
- **DCGAN** (Radford et al., 2015) — *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks* — [Link](https://arxiv.org/abs/1511.06434)
- **WGAN-GP** (Gulrajani et al., 2017) — *Improved Training of Wasserstein GANs* — [Link](https://arxiv.org/abs/1704.00028)
- **StyleGAN** (Karras et al., 2019) — *A Style-Based Generator Architecture for Generative Adversarial Networks* — [Link](https://arxiv.org/abs/1812.04948)
- **DDPM** (Ho et al., 2020) — *Denoising Diffusion Probabilistic Models* — [Link](https://arxiv.org/abs/2006.11239)

### Self-Supervised Learning
- **SimCLR** (Chen et al., 2020) — *A Simple Framework for Contrastive Learning of Visual Representations* — [Link](https://arxiv.org/abs/2002.05709)
- **MoCo** (He et al., 2020) — *Momentum Contrast for Unsupervised Visual Representation Learning* — [Link](https://arxiv.org/abs/1911.05722)
- **MoCo v2** (Chen et al., 2020) — [Link](https://arxiv.org/abs/2003.04297)
- **MAE** (He et al., 2022) — *Masked Autoencoders Are Scalable Vision Learners* — [Link](https://arxiv.org/abs/2111.06377)
- **DINOv2** (Oquab et al., 2023) — *DINOv2: Learning Robust Visual Features without Supervision* — [Link](https://arxiv.org/abs/2304.07193)

## Software & Libraries

| Library | Purpose | Link |
|---------|---------|------|
| **PyTorch** | Deep learning framework | https://pytorch.org |
| **torchvision** | Models, datasets, transforms for CV | https://pytorch.org/vision/stable/ |
| **Torchvision models** | Pre-trained model zoo | https://pytorch.org/vision/stable/models.html |
| **Albumentations** | Fast image augmentation (supports boxes/masks) | https://albumentations.ai |
| **Ultralytics** | YOLOv8 training and inference | https://github.com/ultralytics/ultralytics |
| **Detectron2** | FAIR's detection/segmentation framework | https://github.com/facebookresearch/detectron2 |
| **MMDetection** | OpenMMLab detection toolbox | https://github.com/open-mmlab/mmdetection |
| **Segment Anything** (SAM) | Meta's promptable segmentation | https://github.com/facebookresearch/segment-anything |

## Courses & Tutorials

| Resource | Description |
|----------|-------------|
| **CS231n: Convolutional Neural Networks for Visual Recognition** (Stanford) | The definitive CV course. Covers CNNs, detection, segmentation, GANs, SSL. All lecture notes and assignments online. — [Link](http://cs231n.stanford.edu/) |
| **PyTorch Image Classification Tutorial** | Official tutorial — [Link](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) |
| **Torchvision Object Detection Tutorial** | Official fine-tuning tutorial — [Link](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) |
| **Albumentations Demo** | Augmentation examples — [Link](https://albumentations.ai/docs/examples/) |
| **Papers With Code: Image Classification** | Benchmark leaderboards and code — [Link](https://paperswithcode.com/task/image-classification) |
| **fastai Practical Deep Learning** | Lecun-style top-down teaching, excellent for transfer learning — [Link](https://course.fast.ai/) |

## Datasets

| Dataset | Task | Size | Link |
|---------|------|------|------|
| CIFAR-10 | Classification | 60K images, 10 classes | https://www.cs.toronto.edu/~kriz/cifar.html |
| ImageNet | Classification | 1.2M images, 1K classes | https://www.image-net.org/ |
| COCO | Detection + Segmentation | 330K images, 80 objects, 91 stuff | https://cocodataset.org/ |
| PASCAL VOC | Detection + Segmentation | 11K images, 20 classes | http://host.robots.ox.ac.uk/pascal/VOC/ |
| Cityscapes | Segmentation (urban driving) | 5K fine + 20K coarse, 30 classes | https://www.cityscapes-dataset.com/ |
| Open Images | Detection | 9M images, 600 classes | https://storage.googleapis.com/openimages/web/index.html |

## Visualisation & Interpretability

- **Grad-CAM** — Class Activation Mapping — [Paper](https://arxiv.org/abs/1610.02391) | [PyTorch impl](https://github.com/jacobgil/pytorch-grad-cam)
- **Captum** — PyTorch model interpretability library — [Link](https://captum.ai/)
- **TensorBoard** — Loss curves, histograms, image visualisation — `pip install tensorboard`
- **Weights & Biases (wandb)** — Experiment tracking — [Link](https://wandb.ai/)

## Tips for Practical Work

1. **Start with a simple baseline** — linear classifier on pre-trained features before training a full model.
2. **Use `torchinfo`** for model summaries: `pip install torchinfo; summary(model, (3, 224, 224))`.
3. **Gradient accumulation** for large models on small GPUs:
   ```python
   loss.backward()
   if batch_idx % accumulation_steps == 0:
       optimizer.step()
       optimizer.zero_grad()
   ```
4. **Mixed precision training** for 2× speedup: `from torch.cuda.amp import autocast, GradScaler`.
5. **Debug with a single batch** — overfit one batch first to ensure the model can learn before full training.
