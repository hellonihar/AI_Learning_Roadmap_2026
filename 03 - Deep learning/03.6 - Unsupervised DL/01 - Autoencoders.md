# Autoencoders

## Overview

An autoencoder is a type of artificial neural network trained to learn efficient representations of data in an unsupervised manner. The core idea is deceptively simple: the network learns to copy its input to its output. What makes this useful is the architectural constraint imposed on the network — a bottleneck that forces the model to compress the input into a lower-dimensional latent space and then reconstruct it. By learning to ignore noise and focus on the most salient features, the autoencoder discovers a meaningful compressed representation of the data.

## Encoder-Decoder Architecture

An autoencoder consists of two main components connected by a central bottleneck layer.

**Encoder:** The encoder is a function $f_\phi$ that maps the input $x \in \mathbb{R}^d$ to a latent representation $z \in \mathbb{R}^p$ where typically $p < d$. This is usually implemented as one or more fully connected (dense) layers with non-linear activations:

$$z = f_\phi(x) = \sigma(W_e x + b_e)$$

**Decoder:** The decoder is a function $g_\theta$ that maps the latent representation $z$ back to the original input space, producing a reconstruction $\hat{x} \in \mathbb{R}^d$:

$$\hat{x} = g_\theta(z) = \sigma(W_d z + b_d)$$

The overall autoencoder can be expressed as $\hat{x} = g_\theta(f_\phi(x))$.

**Bottleneck / Latent Space:** The layer where $z$ resides is called the bottleneck. Its dimensionality $p$ controls the amount of compression. When $p < d$, the network must learn to discard redundant information and keep only the essential features needed for reconstruction. This forces the model to discover the underlying low-dimensional manifold of the data.

## Reconstruction Loss

The autoencoder is trained to minimize the reconstruction error between the input $x$ and the reconstruction $\hat{x}$. The most common loss function for continuous data is the **Mean Squared Error (MSE)**:

$$\mathcal{L}(x, \hat{x}) = \frac{1}{n} \sum_{i=1}^{n} \|x_i - \hat{x}_i\|^2$$

For binary data, binary cross-entropy is often used instead. The training objective is to minimize this loss over the training dataset, updating both encoder and decoder parameters jointly via backpropagation.

## Undercomplete vs. Overcomplete Autoencoders

### Undercomplete Autoencoders

An undercomplete autoencoder has a bottleneck dimension $p$ that is smaller than the input dimension $d$. This is the classic autoencoder setup. By constraining the capacity of the network, the model is forced to learn the most important features of the data distribution.

- **Advantage:** Prevents the network from simply copying the input without learning meaningful features.
- **Use case:** Dimensionality reduction, where the latent space $z$ serves as a compressed feature representation analogous to PCA but with non-linear transformations.

### Overcomplete Autoencoders

An overcomplete autoencoder has a bottleneck dimension $p$ that is *larger* than the input dimension $d$. Without additional regularization, the network can trivially learn the identity function, making it useless for representation learning.

- **Regularization needed:** To make overcomplete autoencoders useful, some form of regularization must be applied — weight decay, dropout, or sparsity constraints.
- **Use case:** When paired with sparsity penalties, overcomplete autoencoders can learn overcomplete basis functions (more features than inputs), which is useful for feature discovery.

## Applications

### Dimensionality Reduction

Autoencoders provide a non-linear generalization of PCA. While PCA finds a linear subspace that preserves maximum variance, autoencoders can learn non-linear manifolds. The latent space $z$ serves as the reduced-dimension representation and can be used for visualization (via 2D or 3D bottlenecks), as input features for downstream supervised models, or for compression.

### Denoising

By training the autoencoder to reconstruct clean data from a corrupted version of the input, the model learns to filter out noise. This is covered in detail in the next section (Denoising Autoencoders). The network must learn the underlying structure of the data well enough to distinguish signal from noise.

### Anomaly Detection

Anomaly detection is one of the most successful practical applications of autoencoders. The idea is:

1. Train an autoencoder exclusively on "normal" (non-anomalous) data.
2. For normal data points, the autoencoder reconstructs well — reconstruction error is low.
3. For anomalous data points (not seen during training), the autoencoder fails to reconstruct them accurately — reconstruction error is high.

This approach works because the autoencoder learns the distribution of normal data. Anomalies lie outside this distribution and cannot be compressed and reconstructed effectively. The reconstruction error itself serves as the anomaly score, and a threshold can be set to flag anomalies.

## Practical Considerations

- **Activation functions:** ReLU or leaky ReLU for hidden layers; linear or sigmoid for the output layer depending on data range.
- **Weight initialization:** He or Xavier initialization.
- **Regularization:** L2 weight decay, dropout, or early stopping to prevent overfitting.
- **Batch size:** Typically 32–256; smaller batches act as a regularizer.
- **Optimizer:** Adam is a strong default choice.

## Summary

Autoencoders are foundational unsupervised deep learning models that learn compressed representations via reconstruction. The bottleneck constraint is the key mechanism forcing useful feature learning. Undercomplete autoencoders perform non-linear dimensionality reduction, while overcomplete variants require additional regularization. Practical applications span dimensionality reduction, denoising, and anomaly detection. Understanding autoencoders is essential preparation for more advanced models like variational autoencoders and GANs.
