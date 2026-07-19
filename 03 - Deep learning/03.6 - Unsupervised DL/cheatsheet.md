# Cheatsheet: Unsupervised Deep Learning

---

## Autoencoder Architecture

```
Input x ──► Encoder f_φ ──► Latent z ──► Decoder g_θ ──► Reconstruction x̂
            (compress)          ↑          (expand)
                            Bottleneck
                          (p < d for undercomplete)
```

- **Loss:** $\mathcal{L} = \|x - \hat{x}\|^2$ (MSE)
- **Undercomplete:** $p < d$, forced compression
- **Overcomplete:** $p > d$, needs regularization (sparsity, weight decay)
- **Applications:** Dimensionality reduction, denoising, anomaly detection

---

## VAE: ELBO Loss

$$\mathcal{L}_{\text{VAE}} = -\underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction}} + \underbrace{\text{KL}(q_\phi(z|x) \parallel p(z))}_{\text{Regularization}}$$

- **Encoder outputs:** $\mu$, $\log\sigma^2$
- **Reparameterization:** $z = \mu + \sigma \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$
- **KL closed-form** (Gaussian prior):
  $$\text{KL} = -\frac{1}{2} \sum_{j} \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$
- **$\beta$-VAE:** $\mathcal{L} = -\mathbb{E}[\log p_\theta(x|z)] + \beta \cdot \text{KL}$

---

## GAN: Minimax Objective

$$\min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

- **Non-saturating G loss:** $\max_G \mathbb{E}_z[\log D(G(z))]$
- **Nash equilibrium:** $p_G = p_{\text{data}}$, $D(x) = \frac{1}{2}$
- **Common pitfalls:** Mode collapse, vanishing gradients, non-convergence

---

## GAN Variants Comparison

| Variant | Key Innovation | Training | Use Case |
|---------|---------------|----------|----------|
| cGAN | Conditioning $y$ on $G$ and $D$ | Moderate | Controlled generation |
| WGAN | Wasserstein distance | Better | General generation |
| WGAN-GP | Gradient penalty for Lipschitz | Good | High-quality generation |
| CycleGAN | Cycle consistency loss | Good | Unpaired translation |
| StyleGAN | Mapping network + AdaIN | Good | Controllable high-res |

---

## Diffusion Process

**Forward (fixed):** $q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1}, \beta_t I)$

**Direct sample:** $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$, $\epsilon \sim \mathcal{N}(0, I)$

**Reverse (learned):** $p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t,t), \Sigma_\theta(x_t,t))$

**Training loss:** $\mathcal{L} = \mathbb{E}_{t,x_0,\epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$

**DDIM:** Non-Markovian, faster sampling (10–50 steps vs 1000)

---

## Quick Reference: When to Use What

| Task | Model |
|------|-------|
| Compress data, reduce dimensions | Autoencoder |
| Remove noise from images | Denoising AE |
| Learn interpretable features | Sparse AE |
| Generate new data (smooth latent space) | VAE |
| Generate sharp, realistic images | GAN / StyleGAN |
| Translate images (unpaired) | CycleGAN |
| State-of-the-art image generation | Diffusion Models |
| Anomaly detection | AE / VAE |
