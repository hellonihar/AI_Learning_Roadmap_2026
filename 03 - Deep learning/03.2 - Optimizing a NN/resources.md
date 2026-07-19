# Resources

## Foundational Papers

| Paper | Year | Key Contribution |
|---|---|---|
| [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) — Srivastava et al. | 2014 | Dropout |
| [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) — Ioffe & Szegedy | 2015 | BatchNorm |
| [Layer Normalization](https://arxiv.org/abs/1607.06450) — Ba et al. | 2016 | LayerNorm |
| [Group Normalization](https://arxiv.org/abs/1803.08494) — Wu & He | 2018 | GroupNorm |
| [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Kingma & Ba | 2014 | Adam optimizer |
| [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Loshchilov & Hutter | 2017 | AdamW |
| [Incorporating Nesterov Momentum into Adam](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ) — Dozat | 2016 | Nadam |
| [On the importance of initialization and momentum in deep learning](https://www.cs.toronto.edu/~hinton/absps/momentum.pdf) — Sutskever et al. | 2013 | Nesterov for deep learning |
| [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) — Smith | 2015 | Cyclic LR |
| [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) — Loshchilov & Hutter | 2016 | Cosine annealing + restarts |
| [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) — Bergstra & Bengio | 2012 | Random search |
| [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html) — Bergstra et al. | 2011 | TPE |
| [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846) — Jaderberg et al. | 2017 | PBT |
| [Gradient Clipping](https://arxiv.org/abs/1211.5063) — Pascanu et al. | 2013 | Gradient clipping theory |

## Courses & Lectures

| Resource | Link | Covers |
|---|---|---|
| CS231n (Stanford) — Neural Networks Part 2 | [YouTube](https://www.youtube.com/watch?v=_JB0Cn0zB2o) | Optimizers, regularization, BatchNorm |
| CS231n (Stanford) — Neural Networks Part 3 | [YouTube](https://www.youtube.com/watch?v=vq2nn5WZ0Oo) | Hyperparameter tuning, learning rates |
| Deep Learning Specialization — Course 2 (Coursera) | [coursera.org](https://www.coursera.org/learn/deep-neural-network) | Regularization, optimizers, hyperparameter tuning |
| Fast.ai — Practical Deep Learning | [course.fast.ai](https://course.fast.ai/) | SGD, transfer learning, scheduling |
| NYU Deep Learning (Yann LeCun) | [YouTube](https://www.youtube.com/playlist?list=PLLHTzKZzVU9eauZ2eRTDnF3GoFB4kM6Sf) | Optimization theory |

## Blog Posts & Tutorials

| Title | Link |
|---|---|
| An overview of gradient descent optimization algorithms (Ruder) | [ruder.io/optimizing-gradient-descent](https://ruder.io/optimizing-gradient-descent/) |
| Why Momentum Really Works | [distill.pub/2017/momentum](https://distill.pub/2017/momentum/) |
| Batch Normalization Explained | [paperswithcode.com](https://paperswithcode.com/method/batch-normalization) |
| A Visual Explanation of Gradient Descent Methods | [towardsdatascience.com](https://towardsdatascience.com/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b1020e2) |
| Hyperparameter Optimization — A Practical Guide | [neptune.ai/blog/hyperparameter-tuning](https://neptune.ai/blog/hyperparameter-tuning) |

## Tools

| Tool | Use |
|---|---|
| [Optuna](https://optuna.org/) | Hyperparameter optimization framework |
| [Ray Tune](https://docs.ray.io/en/latest/tune/) | Distributed hyperparameter tuning |
| [Weights & Biases Sweeps](https://docs.wandb.ai/guides/sweeps) | Cloud-based tuning |
| [PyTorch Optimizer Docs](https://pytorch.org/docs/stable/optim.html) | Reference for all optimizers |
| [TensorFlow Optimizer Docs](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) | TF optimizer reference |
