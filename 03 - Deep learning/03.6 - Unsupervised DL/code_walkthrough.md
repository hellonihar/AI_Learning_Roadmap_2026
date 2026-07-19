# Code Walkthrough: Unsupervised Deep Learning with NumPy

This walkthrough implements core unsupervised deep learning concepts using **only NumPy** — no PyTorch, TensorFlow, or JAX. This builds intuition for what these models actually compute under the hood.

---

## 1. Feedforward Autoencoder from Scratch

We implement a simple undercomplete autoencoder for 2D data compression (1D bottleneck).

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def mse_loss(x, x_hat):
    return np.mean((x - x_hat) ** 2)

class Autoencoder:
    def __init__(self, input_dim, hidden_dim):
        self.W_e = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b_e = np.zeros((1, hidden_dim))
        self.W_d = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b_d = np.zeros((1, input_dim))

    def encode(self, x):
        return relu(x @ self.W_e + self.b_e)

    def decode(self, z):
        return z @ self.W_d + self.b_d  # linear output

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def backward(self, x, x_hat, z, lr=0.01):
        m = x.shape[0]
        dL_dx_hat = (x_hat - x) / m  # d(MSE)/dx_hat

        # Decoder gradients
        dL_dW_d = z.T @ dL_dx_hat
        dL_db_d = np.sum(dL_dx_hat, axis=0, keepdims=True)

        # Backprop through encoder
        dL_dz = dL_dx_hat @ self.W_d.T
        dL_dh = dL_dz * relu_deriv(z)  # through ReLU
        dL_dW_e = x.T @ dL_dh
        dL_db_e = np.sum(dL_dh, axis=0, keepdims=True)

        # Gradient descent
        self.W_d -= lr * dL_dW_d
        self.b_d -= lr * dL_db_d
        self.W_e -= lr * dL_dW_e
        self.b_e -= lr * dL_db_e

    def train(self, x, epochs=500, lr=0.01, verbose=True):
        for epoch in range(epochs):
            x_hat, z = self.forward(x)
            loss = mse_loss(x, x_hat)
            self.backward(x, x_hat, z, lr)
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:3d}, Loss: {loss:.6f}")

# Synthetic data: points on a 2D curve
np.random.seed(42)
t = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
x_data = np.hstack([np.sin(t), np.cos(t)]) + 0.02 * np.random.randn(200, 2)

ae = Autoencoder(input_dim=2, hidden_dim=1)
ae.train(x_data, epochs=1000, lr=0.05)

x_recon, z_latent = ae.forward(x_data)
print(f"Final reconstruction loss: {mse_loss(x_data, x_recon):.6f}")
print(f"Latent codes (first 5): {z_latent[:5].ravel()}")
```

### What's happening

- The encoder compresses (x, y) coordinates into a single latent value $z$.
- The decoder reconstructs (x, y) from $z$.
- The bottleneck dimension (1) forces the model to learn the underlying 1D manifold — the circular structure of the data.
- This is analogous to non-linear PCA.

---

## 2. VAE: The Reparameterization Trick (Conceptual)

VAEs require backpropagation through random sampling. The reparameterization trick makes this possible.

```python
def reparameterize(mu, log_var):
    """
    Sample z ~ N(mu, sigma^2) using the reparameterization trick.

    Instead of sampling directly (which is non-differentiable),
    we sample epsilon ~ N(0, I) and compute:  z = mu + sigma * epsilon
    Gradients flow through mu and sigma.
    """
    sigma = np.exp(0.5 * log_var)
    epsilon = np.random.randn(*mu.shape)
    z = mu + sigma * epsilon
    return z

# Example: VAE encoder output
def vae_encoder(x, W_e, b_e):
    h = relu(x @ W_e + b_e)
    mu = h @ W_mu + b_mu           # mean
    log_var = h @ W_var + b_var    # log-variance
    return mu, log_var

# Sample latent code (differentiable)
mu, log_var = vae_encoder(x_data, W_e, b_e)
z = reparameterize(mu, log_var)   # gradients can now flow through mu, log_var

# KL divergence between N(mu, sigma^2) and N(0, I)
kl_loss = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1)
```

### Why it matters

The reparameterization trick is what makes VAE training possible. Without it, the sampling operation would block gradients and we could not train the encoder to produce meaningful latent distributions.

---

## 3. GAN Training Loop (Algorithmic Flow)

Real GANs require DL frameworks for efficient backpropagation through convolutional layers. The algorithmic flow can be represented in NumPy to build intuition.

```python
def gan_training_step(x_real, G, D, z_dim, lr_G, lr_D):
    """
    One GAN training iteration (conceptual).

    G: generator function  G(z; theta_G) -> x_fake
    D: discriminator function  D(x; theta_D) -> probability
    """
    m = x_real.shape[0]

    # --- Train Discriminator ---
    # Sample random noise
    z = np.random.randn(m, z_dim)
    x_fake = G(z)

    # Compute discriminator outputs
    D_real = D(x_real)
    D_fake = D(x_fake)

    # Discriminator loss: -[log D(x) + log(1 - D(G(z)))]
    D_loss = -np.mean(np.log(D_real + 1e-8) + np.log(1 - D_fake + 1e-8))

    # Backprop to update D parameters
    # dD_loss / dtheta_D  (done by DL framework)

    # --- Train Generator ---
    # Sample new noise
    z = np.random.randn(m, z_dim)
    x_fake = G(z)
    D_fake = D(x_fake)

    # Generator loss (non-saturating): -log D(G(z))
    G_loss = -np.mean(np.log(D_fake + 1e-8))

    # Backprop to update G parameters
    # dG_loss / dtheta_G  (done by DL framework)

    return D_loss, G_loss

# Training loop
for step in range(10000):
    x_batch = get_real_batch()       # sample from dataset
    D_loss, G_loss = gan_training_step(x_batch, G, D, z_dim=100, lr_G=2e-4, lr_D=2e-4)

    if step % 1000 == 0:
        print(f"Step {step}, D loss: {D_loss:.4f}, G loss: {G_loss:.4f}")
```

### Key algorithmic insights

1. The discriminator and generator are updated **alternatingly** — not simultaneously.
2. The generator never sees real data; it learns only through the discriminator's feedback.
3. The loss values alone do not indicate convergence. A well-trained GAN has $D(x) \approx D(G(z)) \approx 0.5$.
4. In practice, the discriminator is often trained for several steps per generator step to maintain balance.

---

## Summary

This walkthrough demystifies three core unsupervised deep learning algorithms by implementing their essential components in raw NumPy. The autoencoder shows how compression emerges from reconstruction pressure. The VAE reparameterization trick reveals how stochasticity can be made differentiable. The GAN training loop illustrates the adversarial game that drives generative learning. Understanding these low-level mechanics makes high-level frameworks far more intuitive.
