# Gaussian Mixture Models (GMM)

A **probabilistic** clustering model that assumes data is generated from a mixture of multiple Gaussian distributions.

## How It Works

GMM assigns each point a **probability of belonging to each cluster** (soft assignment), unlike K-Means which gives hard assignments.

A GMM with k components is defined by:

- **Component weights** π₁, π₂, ..., πₖ (how much each Gaussian contributes)
- **Component means** μ₁, μ₂, ..., μₖ (center of each Gaussian)
- **Component covariances** Σ₁, Σ₂, ..., Σₖ (shape of each Gaussian)

## Expectation-Maximization (EM) Algorithm Intuition

GMM is trained using EM. This is an iterative two-step process:

### E-Step (Expectation)
Estimate the probability that each point belongs to each component, given current parameters.

"Given these Gaussians, how likely is this point to come from component 1 vs component 2?"

### M-Step (Maximization)
Update the component parameters to maximize the likelihood of the data, given the current assignments.

"Given these soft assignments, what are the best new means/covariances/weights?"

Repeat until convergence.

EM is a general framework used beyond GMM — for handling missing data, latent variable models, and more.

## Covariance Type

| Type | Shape | Flexibility | Parameters |
|---|---|---|---|
| `full` | Each component has its own covariance | Maximum flexibility | Many (k × d²) |
| `tied` | All components share the same covariance | Moderate | d² |
| `diag` | Each component has independent variances | Limited | k × d |
| `spherical` | Each component has a single variance | Very limited | k |

## GMM vs K-Means

| Aspect | K-Means | GMM |
|---|---|---|
| **Assignment** | Hard (point → one cluster) | Soft (point → probability per cluster) |
| **Cluster shape** | Spherical only | Elliptical (any shape) |
| **Cluster size** | Assumes equal size | Can vary |
| **Output** | Cluster label | Probabilities + label |
| **EM-based?** | No (iterative distance) | Yes |

## Examples

1. **Customer segmentation with overlap**: A customer who shops at both budget and luxury stores isn't "either budget or luxury" — they have a 60% probability of being budget and 40% luxury. GMM captures this ambiguity. Marketing can target based on probabilities: "70%+ likely budget → send discount offers."
2. **Image background subtraction**: Video pixels belong to either "background" or "foreground." GMM models each pixel's color distribution over time (multiple Gaussians for background at different times of day). A pixel with low probability under background GMM is classified as foreground (moving object).
3. **Anomaly detection**: Fit GMM to normal transaction data. Points with very low probability under all components are anomalies. Unlike one-class SVM, GMM naturally handles multiple "normal" modes (e.g., weekday spending vs weekend spending are different Gaussians).
