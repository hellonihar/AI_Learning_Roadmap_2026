# Kernel PCA

An extension of PCA that uses the **kernel trick** to find non-linear structure.

## How It Works

Linear PCA captures variance in the original feature space. Kernel PCA implicitly maps data to a **higher-dimensional space** (via kernel) and performs PCA there. Non-linear patterns in the original space become linear in the higher-dimensional space.

## Kernel Options

| Kernel | Behavior |
|---|---|
| **RBF** | Most common. Captures local non-linear structure. γ controls smoothness. |
| **Polynomial** | Captures polynomial relationships. Degree controls complexity. |
| **Cosine** | Good for text data (direction matters more than magnitude). |
| **Sigmoid** | Similar to neural network activations. |

## When to Use Kernel PCA Over Linear PCA

- Data has clear **non-linear structure** (e.g., concentric circles, Swiss roll)
- Linear PCA doesn't capture the pattern you see in the data
- Visualization of non-linear datasets

## Limitations

- **More hyperparameters** (kernel type, γ, degree) that need tuning
- **Computationally more expensive** than linear PCA
- **Interpretability is worse** — components are in kernel space, not original feature space
- The number of components is limited by the number of samples (not features)

## Examples

1. **Swiss roll data**: Points on a 3D spiral (non-linear). Linear PCA sees no clear structure. Kernel PCA (RBF) unfolds the spiral — the first two components reveal the underlying 2D structure.
2. **Concentric circle separation**: Two classes in concentric circles — linear PCA can't separate them. Kernel PCA projects to a space where the circles become linearly separable.
3. **Handwritten digit denoising**: Digits are non-linear manifolds in pixel space. Kernel PCA captures the digit manifold better than linear PCA, producing cleaner reconstructions after denoising.
