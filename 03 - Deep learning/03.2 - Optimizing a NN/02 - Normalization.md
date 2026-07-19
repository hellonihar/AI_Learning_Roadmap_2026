# Normalization

Normalization techniques stabilize training by re-centering and re-scaling intermediate activations. They address **internal covariate shift** — the change in the distribution of layer inputs as preceding layers' parameters update.

## Batch Normalization (BatchNorm)

Introduced by Ioffe & Szegedy (2015), BatchNorm normalizes activations using the mean and variance of the current **mini-batch**.

### Forward Pass

Given a mini-batch $\mathcal{B} = \{x_1, \dots, x_m\}$ with $m$ examples:

$$ \mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i $$

$$ \sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2 $$

$$ \hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} $$

$$ y_i = \gamma \hat{x}_i + \beta $$

The learnable parameters $\gamma$ (scale) and $\beta$ (shift) allow the layer to undo normalization if needed. The small $\epsilon$ prevents division by zero.

### Backward Pass

Let $\hat{\sigma}^2 = \sigma_{\mathcal{B}}^2 + \epsilon$ and $n = m$ for brevity. The gradients with respect to each input:

$$ \frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{n} \frac{\gamma}{\sqrt{\hat{\sigma}^2}} \left[ n \frac{\partial \mathcal{L}}{\partial y_i} - \sum_{j=1}^{n} \frac{\partial \mathcal{L}}{\partial y_j} - \hat{x}_i \sum_{j=1}^{n} \frac{\partial \mathcal{L}}{\partial y_j} \hat{x}_j \right] $$

The gradients for $\gamma$ and $\beta$:

$$ \frac{\partial \mathcal{L}}{\partial \gamma} = \sum_{i=1}^{n} \frac{\partial \mathcal{L}}{\partial y_i} \hat{x}_i $$

$$ \frac{\partial \mathcal{L}}{\partial \beta} = \sum_{i=1}^{n} \frac{\partial \mathcal{L}}{\partial y_i} $$

### Inference

At test time, the population mean and variance (running averages computed during training) are used:

$$ y = \gamma \frac{x - \mu_{\text{pop}}}{\sqrt{\sigma_{\text{pop}}^2 + \epsilon}} + \beta $$

**Why it works**: BatchNorm smoothes the loss landscape, allowing higher learning rates. It reduces sensitivity to initialization and adds a mild regularization effect (each example is normalized by the batch, introducing noise).

**Pros**: Enables higher LR, reduces need for Dropout, accelerates convergence.
**Cons**: Small batch sizes produce noisy estimates; batch-dependent behavior complicates RNNs and sequence models.

## Layer Normalization (LayerNorm)

LayerNorm normalizes across the **feature dimension** for each sample independently (Ba et al., 2016).

$$ \mu = \frac{1}{d} \sum_{j=1}^{d} x_j, \quad \sigma^2 = \frac{1}{d} \sum_{j=1}^{d} (x_j - \mu)^2 $$

$$ \hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_j = \gamma \hat{x}_j + \beta $$

**When to use**: NLP/Transformer models (each token normalized independently). RNNs, where batch size varies. Stable with any batch size.

## Instance Normalization (InstanceNorm)

InstanceNorm normalizes per channel per sample (Ulyanov et al., 2016). For a tensor $x \in \mathbb{R}^{C \times H \times W}$, compute $\mu$ and $\sigma$ per $(c, n)$ pair.

$$ \mu_{nc} = \frac{1}{HW} \sum_{h,w} x_{nchw}, \quad \sigma_{nc}^2 = \frac{1}{HW} \sum_{h,w} (x_{nchw} - \mu_{nc})^2 $$

**When to use**: Image style transfer. Each image's style is localized in its own statistics — InstanceNorm removes instance-specific contrast information.

## Group Normalization (GroupNorm)

GroupNorm divides channels into $G$ groups and normalizes within each group (Wu & He, 2018). It interpolates between InstanceNorm ($G = C$) and LayerNorm ($G = 1$).

$$ \mu_{ng} = \frac{1}{(C/G)HW} \sum_{\text{group } g} x, \quad \sigma_{ng}^2 = \frac{1}{(C/G)HW} \sum_{\text{group } g} (x - \mu_{ng})^2 $$

**When to use**: When batch size is small (e.g., video, medical imaging, object detection). GroupNorm with $G = 32$ or $G = 16$ is a strong drop-in replacement for BatchNorm at low batch sizes.

## Comparison Table

| Norm | Normalization axis | Batch-dependent | Good for |
|---|---|---|---|
| BatchNorm | Batch $\times$ spatial | Yes | CNNs with large batches |
| LayerNorm | Feature | No | Transformers, RNNs |
| InstanceNorm | Spatial (per channel) | No | Style transfer |
| GroupNorm | Group of channels $\times$ spatial | No | CNNs with small batches |

## Which One to Use?

- **CNNs with large batches**: BatchNorm (default).
- **CNNs with small batches ($\leq 8$)**: GroupNorm (often $G=32$).
- **Transformers**: LayerNorm (always).
- **RNNs/LSTMs**: LayerNorm.
- **Style transfer / GANs**: InstanceNorm.
- **When batch size varies widely**: LayerNorm or GroupNorm (batch-independent).
