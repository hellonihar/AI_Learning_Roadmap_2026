# Variational Autoencoders (VAE)

## From Deterministic to Probabilistic

The standard autoencoder learns a deterministic mapping from input $x$ to latent code $z$ and back. This works well for compression and reconstruction, but it has a major limitation: the latent space is not regularized. Points far from the training data in latent space may decode to meaningless outputs. Variational Autoencoders (VAEs), introduced by Kingma and Welling in 2013, fix this by learning a **probabilistic** mapping where the encoder outputs a probability distribution over the latent space rather than a single point.

## The Probabilistic Framework

A VAE models the data generation process as follows:
- A latent variable $z$ is drawn from a prior distribution $p(z)$ (typically a standard Gaussian $\mathcal{N}(0, I)$).
- The observed data $x$ is generated from a conditional distribution $p_\theta(x|z)$ (the decoder).

The goal is to learn the model parameters $\theta$ that maximize the marginal likelihood $p_\theta(x) = \int p_\theta(x|z) p(z) \, dz$.

However, the integral over $z$ is intractable. We introduce an approximate posterior $q_\phi(z|x)$ (the encoder) to approximate the true posterior $p_\theta(z|x)$.

## The Encoder: Outputting Distribution Parameters

Unlike a standard autoencoder that outputs a single latent vector $z$, the VAE encoder outputs the parameters of a Gaussian distribution:

$$\mu = f_\phi^\mu(x), \quad \log\sigma^2 = f_\phi^\sigma(x)$$

The latent variable is then sampled as $z \sim \mathcal{N}(\mu, \sigma^2 I)$.

## The Reparameterization Trick

Sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ is a stochastic operation that blocks gradients from flowing back to the encoder. The reparameterization trick separates the stochasticity from the learnable parameters:

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Now, gradients can flow through $\mu$ and $\sigma$ because the sampling is moved to the auxiliary noise variable $\epsilon$, which is independent of the model parameters. This allows end-to-end training via standard backpropagation.

## The ELBO Derivation

We want to maximize the log marginal likelihood $\log p_\theta(x)$. Using the evidence lower bound (ELBO):

$$\log p_\theta(x) = \text{KL}(q_\phi(z|x) \parallel p_\theta(z|x)) + \mathcal{L}_{\text{ELBO}}$$

Since KL divergence is non-negative, $\mathcal{L}_{\text{ELBO}}$ is a lower bound on $\log p_\theta(x)$. Deriving $\mathcal{L}_{\text{ELBO}}$:

$$
\begin{aligned}
\log p_\theta(x) &= \log \int p_\theta(x|z) p(z) \, dz \\
&= \log \int p_\theta(x|z) \frac{p(z)}{q_\phi(z|x)} q_\phi(z|x) \, dz \\
&\geq \int q_\phi(z|x) \log \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)} \, dz \quad \text{(Jensen's inequality)} \\
&= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \parallel p(z))
\end{aligned}
$$

Thus, the ELBO loss (negated for minimization) is:

$$\mathcal{L}_{\text{VAE}} = -\underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction Term}} + \underbrace{\text{KL}(q_\phi(z|x) \parallel p(z))}_{\text{KL Divergence}}$$

### Reconstruction Term

This measures how well the decoder reconstructs the input from the latent code. For real-valued data, $\log p_\theta(x|z)$ is typically a Gaussian log-likelihood, which reduces to MSE:

$$\log p_\theta(x|z) \propto -\|x - \hat{x}\|^2$$

For binary data, it becomes binary cross-entropy.

### KL Divergence Term

This regularizes the encoder to keep the latent distribution close to the prior $p(z) = \mathcal{N}(0, I)$. It has a closed-form solution when both $q_\phi(z|x)$ and $p(z)$ are Gaussian:

$$
\text{KL}(\mathcal{N}(\mu, \sigma^2) \parallel \mathcal{N}(0, I)) = -\frac{1}{2} \sum_{j=1}^{J} \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)
$$

This term penalizes the encoder when:
- The mean $\mu$ deviates from 0.
- The variance $\sigma^2$ deviates from 1.

The KL term acts as a regularizer, organizing the latent space into a smooth, continuous manifold that is amenable to interpolation and generation.

## Generative Capability

The trained VAE can generate new data points by:
1. Sampling $z \sim \mathcal{N}(0, I)$ from the prior.
2. Passing $z$ through the decoder to produce $\hat{x} = g_\theta(z)$.

Because the KL term forces the latent distributions of all training points to overlap with the standard Gaussian prior, every point in the latent space decodes to a plausible output. This enables:

- **Interpolation:** Linearly interpolating between two latent codes produces smooth transitions in the output space.
- **Controlled generation:** Manipulating specific dimensions of $z$ (if they become disentangled) controls semantic attributes of the output.
- **Density estimation:** The ELBO provides a lower bound on the likelihood of new data points.

## The Reconstruction vs. Regularization Trade-off

The VAE loss involves a trade-off controlled by the weight $\beta$ (introduced in $\beta$-VAE):

$$\mathcal{L}_{\beta\text{-VAE}} = -\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + \beta \cdot \text{KL}(q_\phi(z|x) \parallel p(z))$$

- $\beta = 1$: Standard VAE.
- $\beta > 1$: Stronger regularization, leading to more disentangled representations but lower reconstruction quality.
- $\beta < 1$: Weaker regularization, better reconstructions but less structured latent space.

## Practical Considerations

- **KL annealing:** Gradually increasing the KL weight during training can help avoid posterior collapse (where the KL term dominates and the latent code is ignored).
- **Posterior collapse:** When the decoder is too powerful (e.g., an autoregressive decoder for images), it may learn to ignore $z$ entirely, reducing the VAE to a standard autoencoder.
- **Free bits:** A technique that ensures a minimum amount of KL divergence per latent dimension to prevent collapse.

## Summary

VAEs bring probabilistic reasoning to autoencoders by learning a distribution over the latent space. The reparameterization trick enables gradient-based optimization of the ELBO, which balances reconstruction fidelity with latent space regularization. VAEs are powerful generative models with smooth latent spaces, making them ideal for interpolation, controlled generation, and representation learning. They form a bridge between traditional autoencoders and modern diffusion models.
