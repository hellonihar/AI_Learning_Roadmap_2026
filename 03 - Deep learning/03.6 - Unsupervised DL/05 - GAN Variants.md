# GAN Variants

## Conditional GAN (cGAN)

The Conditional GAN extends the standard GAN by conditioning both the generator and discriminator on additional information $y$ (a class label, text embedding, or another modality).

**Objective:**

$$
\min_{G} \max_{D} V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x|y)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z|y)|y))]
$$

The conditioning information $y$ is fed as an additional input to both networks. For the generator, $y$ is concatenated with the noise vector $z$. For the discriminator, $y$ is concatenated with the input $x$ (or with a hidden representation).

**Key innovation:** cGANs enable controlled generation — specifying the class of the output (e.g., generate a "cat" vs. a "dog"). This was foundational for later advances in text-to-image synthesis (StackGAN, AttnGAN) and image-to-image translation (Pix2Pix).

## Wasserstein GAN (WGAN) & WGAN-GP

The Wasserstein GAN (Arjovsky et al., 2017) addresses the fundamental issue of vanishing gradients by changing the loss function from JS divergence to the Wasserstein-1 distance (Earth Mover's distance).

**Standard GAN problem:** When the real and generated distributions have disjoint supports (common in high-dimensional spaces), JS divergence is constant and gradients vanish. The Wasserstein distance provides meaningful gradients even when distributions do not overlap.

**WGAN formulation:**

$$\min_{G} \max_{\|D\|_L \leq K} \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

The discriminator (now called a "critic") is constrained to be $K$-Lipschitz continuous. In the original WGAN, this was enforced via weight clipping — but this led to optimization difficulties.

**WGAN-GP (Gradient Penalty):** Gulrajani et al. (2017) replaced weight clipping with a gradient penalty term that enforces the Lipschitz constraint directly:

$$\mathcal{L}_{\text{GP}} = \lambda \cdot \mathbb{E}_{\hat{x} \sim p_{\hat{x}}} \left[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2\right]$$

where $\hat{x}$ is sampled uniformly along straight lines between real and generated data points.

**Benefits:** WGAN-GP provides stable training, meaningful loss curves correlated with sample quality, and reduced mode collapse.

## CycleGAN

CycleGAN (Zhu et al., 2017) enables unpaired image-to-image translation — transforming images from one domain to another without requiring paired training examples (e.g., photos to Monet paintings, zebras to horses).

**Architecture:** Two generators ($G: X \to Y$ and $F: Y \to X$) and two discriminators ($D_Y$ for domain $Y$, $D_X$ for domain $X$).

**Cycle consistency loss:** The key innovation enforcing that translations are invertible:

$$\mathcal{L}_{\text{cyc}}(G, F) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\|F(G(x)) - x\|_1] + \mathbb{E}_{y \sim p_{\text{data}}(y)}[\|G(F(y)) - y\|_1]$$

The full objective combines adversarial losses (for domain matching) with cycle consistency losses (for content preservation).

**Applications:** Style transfer, object transfiguration, season translation, photo enhancement, and medical image modality conversion.

## StyleGAN

StyleGAN (Karras et al., 2019) introduces a style-based generator architecture that provides unprecedented control over the generated image at different levels of detail.

**Key innovations:**

1. **Mapping network:** Instead of feeding $z$ directly to the generator, a mapping network $f: z \to w$ transforms the latent code into an intermediate latent space $W$ that is less entangled.

2. **Adaptive Instance Normalization (AdaIN):** The style information from $w$ is injected at each convolutional layer via AdaIN:
   $$\text{AdaIN}(x_i, y) = y_{s,i} \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}$$

3. **Style mixing:** Mixing two different latent codes at different layers allows control over coarse (pose, shape), medium (facial features), and fine (texture, color) details independently.

4. **Stochastic variation:** Noise is added at each layer to generate stochastic details (freckles, hair strands) without affecting the overall composition.

**Impact:** StyleGAN and its successors (StyleGAN2, StyleGAN3) set the state of the art in face generation and enabled semantic editing of real images through inversion into the $W$ space.

## Comparison Table

| Variant | Key Innovation | Training Stability | Use Case |
|---------|---------------|-------------------|----------|
| cGAN | Conditioning on auxiliary info | Moderate | Controlled generation |
| WGAN | Wasserstein distance | Good | General image generation |
| WGAN-GP | Gradient penalty | Very good | High-quality image generation |
| CycleGAN | Cycle consistency | Good | Unpaired image translation |
| StyleGAN | Style-based generator | Good | High-resolution controllable generation |

## Summary

GAN variants have addressed many of the limitations of the original GAN framework. cGANs introduced conditional control, WGAN/WGAN-GP stabilized training via better loss functions, CycleGAN enabled unpaired translation, and StyleGAN provided unprecedented control over generation. Each variant has driven the field forward and expanded the practical applications of adversarial learning.
