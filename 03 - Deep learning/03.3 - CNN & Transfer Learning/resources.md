# Resources: CNN & Transfer Learning

## Essential Papers

### Foundational Architectures

| Paper | Year | Citation | Link |
|---|---|---|---|
| **LeNet-5**: *Gradient-Based Learning Applied to Document Recognition* | 1998 | LeCun et al. | [DOI: 10.1109/5.726791](https://doi.org/10.1109/5.726791) |
| **AlexNet**: *ImageNet Classification with Deep Convolutional Neural Networks* | 2012 | Krizhevsky, Sutskever, Hinton | [NeurIPS 2012](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) |
| **VGG**: *Very Deep Convolutional Networks for Large-Scale Image Recognition* | 2014 | Simonyan, Zisserman | [arXiv:1409.1556](https://arxiv.org/abs/1409.1556) |
| **GoogLeNet (Inception v1)**: *Going Deeper with Convolutions* | 2015 | Szegedy et al. | [CVPR 2015](https://doi.org/10.1109/CVPR.2015.7298594) |
| **ResNet**: *Deep Residual Learning for Image Recognition* | 2016 | He et al. | [CVPR 2016](https://doi.org/10.1109/CVPR.2016.90) |
| **DenseNet**: *Densely Connected Convolutional Networks* | 2017 | Huang et al. | [CVPR 2017](https://doi.org/10.1109/CVPR.2017.243) |

### Modern Advances

| Paper | Year | Citation | Link |
|---|---|---|---|
| **MobileNet**: *Efficient Convolutional Neural Networks for Mobile Vision* | 2017 | Howard et al. | [arXiv:1704.04861](https://arxiv.org/abs/1704.04861) |
| **EfficientNet**: *Rethinking Model Scaling for CNNs* | 2019 | Tan, Le | [ICML 2019](http://proceedings.mlr.press/v97/tan19a.html) |
| **ViT**: *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale* | 2021 | Dosovitskiy et al. | [ICLR 2021](https://openreview.net/forum?id=YicbFdNTTy) |
| **ConvNeXt**: *A ConvNet for the 2020s* | 2022 | Liu et al. | [CVPR 2022](https://doi.org/10.1109/CVPR52688.2022.01067) |

### Visualization & Understanding

| Paper | Year | Citation | Link |
|---|---|---|---|
| *Visualizing and Understanding Convolutional Networks* | 2014 | Zeiler, Fergus | [ECCV 2014](https://doi.org/10.1007/978-3-319-10590-1_53) |
| *Network in Network* (1×1 convs, GAP) | 2013 | Lin et al. | [arXiv:1312.4400](https://arxiv.org/abs/1312.4400) |

### Transfer Learning & Domain Adaptation

| Paper | Year | Citation | Link |
|---|---|---|---|
| *How transferable are features in deep neural networks?* | 2014 | Yosinski et al. | [NeurIPS 2014](https://papers.nips.cc/paper_files/paper/2014/hash/375c71349b295fbe2fcd05092038f7a-Abstract.html) |
| *Domain-Adversarial Training of Neural Networks* | 2016 | Ganin et al. | [JMLR](https://jmlr.org/papers/v17/15-239.html) |

### Data Augmentation

| Paper | Year | Citation | Link |
|---|---|---|---|
| **CutOut**: *Improved Regularization of CNNs with Cutout* | 2017 | DeVries, Taylor | [arXiv:1708.04552](https://arxiv.org/abs/1708.04552) |
| **MixUp**: *Beyond Empirical Risk Minimization* | 2018 | Zhang et al. | [ICLR 2018](https://openreview.net/forum?id=r1Ddp1-Rb) |
| **CutMix**: *Regularization Strategy to Train Strong Classifiers with Localizable Features* | 2019 | Yun et al. | [ICCV 2019](https://doi.org/10.1109/ICCV.2019.00612) |
| **RandAugment**: *Practical Automated Data Augmentation with a Reduced Search Space* | 2020 | Cubuk et al. | [NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/d85b63ef0ccb114d0a3bb7b7d808028f-Abstract.html) |

## Online Courses

| Course | Instructor | Platform | Relevance |
|---|---|---|---|
| **CS231n: CNNs for Visual Recognition** | Fei-Fei Li, Justin Johnson, Serena Yeung | Stanford / YouTube | The definitive CNN course. Covers architectures, backprop, transfer learning, and more. |
| **Deep Learning Specialization (Course 4: CNNs)** | Andrew Ng | Coursera / deeplearning.ai | Excellent introduction with intuitive explanations. |
| **Fast.ai Practical Deep Learning** | Jeremy Howard | fast.ai | Top-down approach — implement state-of-the-art CNNs quickly. |
| **MIT 6.S191: Intro to Deep Learning** | Alexander Amini | MIT | Covers CNNs, RNNs, Transformers in compact lectures. |

## Blog Posts & Tutorials

| Title | Author | Link |
|---|---|---|
| *An Intuitive Explanation of Convolutional Neural Networks* | Ujjwal Karn | [the|datageek blog](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/) |
| *Understanding the Receptive Field of CNNs* | Dang Ha The Hien | [Medium](https://medium.com/@hien.vo12/understanding-the-receptive-field-of-convolutional-neural-networks-3d0373e1e91d) |
| *A Comprehensive Guide to Convolutional Neural Networks* | Sumit Saha | [Towards Data Science](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) |
| *ResNet explained* | Vincent Fung | [Towards Data Science](https://towardsdatascience.com/resnet-28b3fcf80937) |
| *The Illustrated Transformer* (ViT precursor) | Jay Alammar | [jalammar.github.io](https://jalammar.github.io/illustrated-transformer/) |
| *A Walkthrough of Convolution Arithmetic* | Dumoulin, Visin | [arXiv:1603.07285](https://arxiv.org/abs/1603.07285) — excellent reference for conv math |

## Libraries & Tools

| Library | Purpose | Link |
|---|---|---|
| PyTorch | DL framework with built-in CNN modules | [pytorch.org](https://pytorch.org) |
| TensorFlow / Keras | DL framework, high-level API | [tensorflow.org](https://tensorflow.org) |
| torchvision | Pre-trained models, datasets, transforms | [PyTorch Vision](https://pytorch.org/vision/) |
| Albumentations | Fast image augmentation | [albumentations.ai](https://albumentations.ai) |
| Kornia | Differentiable CV on GPU | [kornia.readthedocs.io](https://kornia.readthedocs.io) |
| CNN Explainer | Interactive CNN visualization | [poloclub.github.io/cnn-explainer](https://poloclub.github.io/cnn-explainer/) |
| NN-SVG | Publication-ready NN diagrams | [alexlenail.me](https://alexlenail.me/NN-SVG/) |

## Reference Books

| Title | Author | Relevance |
|---|---|---|
| *Deep Learning* (Ch. 9) | Goodfellow, Bengio, Courville | MIT Press — rigorous treatment of convolution |
| *Deep Learning for Computer Vision* | Rajalingappa, Shanmugamani | Practical guide with Keras implementations |
| *Computer Vision: Algorithms and Applications* | Szeliski | Comprehensive CV reference (broader than DL) |

## Datasets for Practice

| Dataset | Size | Task | Link |
|---|---|---|---|
| MNIST | 70K (28×28) | Digit classification | [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/) |
| CIFAR-10 / CIFAR-100 | 60K (32×32) | Object classification | [cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html) |
| ImageNet | 1.2M (224×224) | 1000-class classification | [image-net.org](https://www.image-net.org) |
| Tiny ImageNet | 110K (64×64) | 200-class classification | [tiny-imagenet](https://tiny-imagenet.herokuapp.com) |
| Stanford Dogs | 20K | Fine-grained classification | [vision.stanford.edu](http://vision.stanford.edu/aditya86/ImageNetDogs/) |
| MedMNIST | Various | Medical image classification | [medmnist.com](https://medmnist.com) |
