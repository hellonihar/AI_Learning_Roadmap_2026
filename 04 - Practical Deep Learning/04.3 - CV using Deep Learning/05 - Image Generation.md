# Image Generation

## The Generative Modelling Problem

Image generation aims to learn the true data distribution `p_data(x)` and sample new images from it. Among the first deep learning approaches to produce realistic images were **Generative Adversarial Networks (GANs)** . A GAN pits two networks against each other: a **Generator** that creates fake images and a **Discriminator** that distinguishes real from fake. The equilibrium state produces a generator capable of generating convincing samples.

## DCGAN: Deep Convolutional GAN

DCGAN (Radford et al., 2015) established architectural guidelines for stable GAN training with convolutions.

### Generator Architecture

The generator maps a latent vector `z ~ N(0, I)` (typically 100-dimensional) to an RGB image. It uses **transposed convolutions** (also called deconvolutions) to upsample spatial dimensions progressively.

```
Input: z (100-dim noise)
   │
Linear(100, 512 × 4 × 4) → Reshape → (512, 4, 4)
   │
ConvTranspose2d(512, 256, 4, stride=2, padding=1) → BatchNorm → ReLU → (256, 8, 8)
   │
ConvTranspose2d(256, 128, 4, stride=2, padding=1) → BatchNorm → ReLU → (128, 16, 16)
   │
ConvTranspose2d(128, 64, 4, stride=2, padding=1) → BatchNorm → ReLU → (64, 32, 32)
   │
ConvTranspose2d(64, 3, 4, stride=2, padding=1) → Tanh → (3, 64, 64)
```

Key details:
- **Transposed convolution** with stride 2 doubles spatial dimensions.
- **BatchNorm** in all generator layers except the output.
- **Tanh** activation at the output scales pixels to `[-1, 1]` (use `Normalize(0.5, 0.5)` for inputs).
- No fully-connected layers beyond the initial projection.

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)
```

### Discriminator Architecture

A standard CNN classifier that outputs a single probability (real vs fake):

```
Input: (3, 64, 64)
   │
Conv2d(3, 64, 4, stride=2, padding=1) → LeakyReLU(0.2) → (64, 32, 32)
   │
Conv2d(64, 128, 4, 2, 1) → BatchNorm → LeakyReLU(0.2) → (128, 16, 16)
   │
Conv2d(128, 256, 4, 2, 1) → BatchNorm → LeakyReLU(0.2) → (256, 8, 8)
   │
Conv2d(256, 1, 4, 1, 0) → Sigmoid → (1,)  # real or fake
```

- Uses **LeakyReLU** (slope 0.2) instead of ReLU to avoid dying gradients.
- **BatchNorm** in all discriminator layers except the input.
- **Strided convolutions** (not pooling) for downsampling.

## Training a DCGAN

### Standard GAN Loss (Binary Cross-Entropy)

```
L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
L_G = -E[log D(G(z))]
```

```python
criterion = nn.BCELoss()

for epoch in range(epochs):
    for real_imgs, _ in dataloader:
        batch = real_imgs.size(0)
        real_labels = torch.ones(batch, 1)
        fake_labels = torch.zeros(batch, 1)

        # Train Discriminator
        z = torch.randn(batch, latent_dim)
        fake_imgs = generator(z)
        d_real = discriminator(real_imgs)
        d_fake = discriminator(fake_imgs.detach())
        d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
        d_loss.backward()
        optim_d.step()

        # Train Generator
        z = torch.randn(batch, latent_dim)
        fake_imgs = generator(z)
        g_loss = criterion(discriminator(fake_imgs), real_labels)
        g_loss.backward()
        optim_g.step()
```

### Training Tips

1. **Label smoothing** — use `0.9` for real labels instead of `1.0` to prevent the discriminator from being overconfident:
   ```python
   real_labels = torch.full((batch, 1), 0.9)
   ```

2. **Balance G/D training** — if the generator loss drops too fast, train the discriminator more (every 2 steps) and vice versa. Monitor both losses — if one dominates, training has collapsed.

3. **Noisy labels** — add small random noise to labels (`real_labels += 0.05 * torch.randn_like(real_labels)`) to improve robustness.

4. **Use Adam** with `lr=0.0002` and `betas=(0.5, 0.999)` — the 0.5 beta1 is critical for GAN stability (lower momentum prevents oscillations).

5. **Batch size** matters — 128 is a good default. Smaller batches destabilise training.

6. **Monitor visual quality** — loss curves alone are unreliable indicators. Save generated samples every epoch and inspect them.

## Beyond DCGAN

| Model | Key Innovation |
|-------|---------------|
| **DCGAN** | Architectural guidelines for stable GANs |
| **WGAN** | Wasserstein loss + weight clipping → no mode collapse |
| **WGAN-GP** | Gradient penalty instead of weight clipping |
| **StyleGAN** | Style-based generator, stochastic variation, extremely high quality |
| **Diffusion Models** | DDPM, Stable Diffusion — current state-of-the-art, reverse diffusion process |

While GANs produce sharp images, they suffer from **mode collapse** (generator produces limited varieties). Diffusion models have largely superseded GANs for image generation at scale, but GANs remain relevant for real-time and conditional generation tasks.

## Summary

DCGAN provides the foundational framework for deep learning-based image generation. The generator uses transposed convolutions to upsample noise into images; the discriminator is a standard CNN. Stable training requires careful balancing, label smoothing, and the Adam optimiser with reduced momentum. Understanding GANs is essential before moving to more advanced generative models like diffusion models.
