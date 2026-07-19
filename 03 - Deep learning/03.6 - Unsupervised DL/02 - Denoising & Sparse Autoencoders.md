# Denoising & Sparse Autoencoders

## Denoising Autoencoders (DAE)

A denoising autoencoder extends the standard autoencoder by corrupting the input and training the network to reconstruct the original, uncorrupted data. This forces the model to learn robust features that capture the true underlying data distribution rather than simply memorizing the identity function.

### Architecture and Training

The training process for a DAE proceeds as follows:

1. **Corrupt the input:** Given an input $x$, we generate a corrupted version $\tilde{x}$ by adding noise (e.g., Gaussian noise) or randomly masking a fraction of the input dimensions (dropout noise).
2. **Encode and decode:** The corrupted input $\tilde{x}$ is passed through the encoder to produce $z = f_\phi(\tilde{x})$, and then through the decoder to produce $\hat{x} = g_\theta(z)$.
3. **Reconstruction loss:** The loss is computed between the reconstruction $\hat{x}$ and the *original* (uncorrupted) input $x$:
   $$\mathcal{L}(x, \hat{x}) = \|x - \hat{x}\|^2$$

### Why It Works

By learning to map corrupted inputs back to clean outputs, the autoencoder cannot simply copy its input. It must learn the statistical dependencies between dimensions — it needs to infer the missing or corrupted parts from the uncorrupted ones. This is akin to learning the data manifold: the model internalizes what kinds of patterns are "natural" and which are not.

### Applications

- **Image denoising:** Removing noise from images (e.g., salt-and-pepper noise, Gaussian blur).
- **Feature learning:** The learned latent representations serve as robust pre-trained features for downstream tasks.
- **Pre-training for Deep Networks:** Stacked denoising autoencoders were historically used for layer-wise pre-training of deep architectures before the advent of better initialization and normalization techniques.

## Sparse Autoencoders (SAE)

A sparse autoencoder adds a sparsity constraint on the activations of the hidden layer. Even with an overcomplete bottleneck ($p > d$), the model is forced to learn useful features because only a fraction of the hidden units are active at any given time.

### Sparsity Penalty

The total loss for a sparse autoencoder combines the reconstruction loss with a sparsity penalty:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{reconstruction}} + \lambda \cdot \Omega_{\text{sparsity}}$$

where $\lambda$ controls the strength of the regularization.

#### L1 Regularization

A simple approach penalizes the absolute value of the hidden activations:

$$\Omega_{\text{sparsity}} = \sum_j |a_j|$$

where $a_j$ is the activation of hidden unit $j$. The L1 penalty encourages many units to have activations close to zero.

#### KL Divergence Sparsity

A more sophisticated approach, popularized by Andrew Ng's group, encourages the average activation of each hidden unit $\hat{\rho}_j$ to match a small target sparsity parameter $\rho$ (e.g., 0.05). The penalty is the KL divergence between a Bernoulli random variable with mean $\rho$ and one with mean $\hat{\rho}_j$:

$$\Omega_{\text{sparsity}} = \sum_j \text{KL}(\rho \parallel \hat{\rho}_j) = \sum_j \left[\rho \log \frac{\rho}{\hat{\rho}_j} + (1 - \rho) \log \frac{1 - \rho}{1 - \hat{\rho}_j}\right]$$

where $\hat{\rho}_j = \frac{1}{m} \sum_{i=1}^{m} a_j^{(i)}$ is the average activation of unit $j$ over the training batch of size $m$.

### Why Sparsity Works

Neuroscientific inspiration: biological neurons exhibit sparse firing patterns — only a small fraction of neurons are active for any given stimulus. In artificial networks, sparsity creates a disentangled representation where individual hidden units specialize in detecting specific features. Different input patterns activate different subsets of the hidden layer, leading to a more interpretable and efficient representation.

### Applications

- **Feature learning on large datasets:** Sparse autoencoders excel at learning overcomplete feature dictionaries (e.g., learning Gabor-like filters from image patches).
- **Pre-training:** Layer-wise pre-training with sparse autoencoders initializes deep networks with meaningful features.
- **Interpretability:** The sparse nature of the activations makes it easier to understand what each hidden unit represents.

## Comparison

| Aspect | Denoising AE | Sparse AE |
|--------|-------------|-----------|
| Constraint | Input corruption | Activation penalty |
| Goal | Robust feature learning | Selective feature activation |
| Bottleneck | Can be undercomplete or overcomplete | Often overcomplete |
| Regularization | Noise injection | Sparsity penalty ($\lambda$) |
| Primary use | Denoising, pre-training | Feature discovery, interpretability |

## Summary

Denoising and sparse autoencoders extend the basic autoencoder framework to learn more robust and meaningful representations. DAEs use input corruption to force the model to understand data structure, while SAEs use activation penalties to encourage specialized feature detectors. Both techniques remain relevant for pre-training, feature learning, and as building blocks for more complex unsupervised architectures.
