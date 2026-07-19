# Optimizers

Optimizers govern how a neural network updates its parameters based on computed gradients. The choice of optimizer dramatically affects convergence speed, final accuracy, and training stability.

## Stochastic Gradient Descent (SGD)

The foundation of deep learning optimization.

$$ w_{t+1} = w_t - \eta \nabla \mathcal{L}(w_t) $$

Where $\eta$ is the learning rate and $\nabla \mathcal{L}(w_t)$ is the gradient of the loss w.r.t. weights at step $t$.

**Intuition**: Take a step in the direction of steepest descent on the current mini-batch. Each step is noisy due to batch sampling.

**Pros**: Simple, generalizes well, well-understood. Less memory than adaptive methods.
**Cons**: Sensitive to learning rate, slow convergence on ill-conditioned problems, struggles with saddle points.

## SGD + Momentum

Accumulates a velocity vector to smooth updates and accelerate in consistent directions (Polyak, 1964).

$$ v_{t+1} = \beta v_t + \nabla \mathcal{L}(w_t) $$
$$ w_{t+1} = w_t - \eta v_{t+1} $$

Common $\beta = 0.9$.

**Intuition**: Ball rolling down a hill — accumulates speed in directions of consistent gradient, dampens oscillations. Momentum helps escape local minima and plateaus by carrying velocity through flat regions.

## Nesterov Accelerated Gradient (NAG)

A lookahead variant of momentum (Nesterov, 1983).

$$ v_{t+1} = \beta v_t + \nabla \mathcal{L}(w_t - \beta \eta v_t) $$
$$ w_{t+1} = w_t - \eta v_{t+1} $$

**Intuition**: Compute gradient at the approximate future position $w_t - \beta \eta v_t$ rather than current position. This "peek ahead" acts as a correction mechanism, preventing overshooting and providing faster convergence than standard momentum.

**Pros**: Faster than standard momentum, better at navigating valleys.
**Cons**: Slightly more computation per step.

## AdaGrad

Adapts per-parameter learning rates based on historical gradient magnitudes (Duchi et al., 2011).

$$ G_t = G_{t-1} + (\nabla \mathcal{L}(w_t))^2 $$
$$ w_{t+1} = w_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot \nabla \mathcal{L}(w_t) $$

**Intuition**: Frequently updated parameters get smaller learning rates; infrequent ones get larger rates. Good for sparse data.

**Pros**: No learning rate tuning per dimension, works well for sparse features (NLP, embeddings).
**Cons**: $G_t$ grows monotonically, causing learning rates to shrink to zero and stop training prematurely.

## RMSProp

Addresses AdaGrad's diminishing LR by using an exponentially decaying average of squared gradients (Hinton, 2012).

$$ v_t = \beta v_{t-1} + (1-\beta) (\nabla \mathcal{L}(w_t))^2 $$
$$ w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \odot \nabla \mathcal{L}(w_t) $$

Typical $\beta = 0.9$, $\epsilon = 10^{-8}$.

**Intuition**: RMSProp adapts the LR inversely to the recent RMS of gradients. It's like AdaGrad with a moving window — the LR adapts but doesn't vanish.

**Pros**: Works well for non-stationary objectives (RNNs), handles varying gradient scales.
**Cons**: Sensitive to $\beta$ and $\eta$ choice.

## Adam (Adaptive Moment Estimation)

Combines momentum with per-parameter adaptive LR (Kingma & Ba, 2014).

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) \nabla \mathcal{L}(w_t) \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) (\nabla \mathcal{L}(w_t))^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
w_{t+1} &= w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

Default: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

**Intuition**: Adam tracks the first moment (mean) and second moment (uncentered variance) of gradients. Bias correction compensates for initialization at zero. The effective step size is approximately $\pm \eta$ in well-conditioned directions and smaller in noisy ones.

**Pros**: Fast convergence, robust to $\eta$ choice, works out-of-the-box for most problems. Handles sparse gradients well.
**Cons**: Can generalize worse than SGD + Momentum, sometimes converges to sharper minima. Sensitive to $\beta_2$.

## AdamW

Adam with decoupled weight decay (Loshchilov & Hutter, 2017).

$$ w_{t+1} = w_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda w_t \right) $$

Key insight: In standard Adam, L2 regularization interacts poorly with the adaptive LR. By decoupling weight decay from gradient updates, AdamW achieves better generalization while maintaining Adam's fast convergence.

**When to use**: Default choice for most modern deep learning (Transformers, CNNs). Replaces Adam in most cases.

## Nadam

Adam with Nesterov momentum incorporated (Dozat, 2016).

Nadam replaces the momentum term $\hat{m}_t$ with the Nesterov lookahead:

$$ \hat{m}_t = \frac{\beta_1 m_{t-1} + (1-\beta_1) \nabla \mathcal{L}(w_t)}{1 - \beta_1^t} $$
$$ \tilde{m}_t = (1-\beta_1) \hat{m}_t + \beta_1 \hat{m}_{t+1} $$
$$ w_{t+1} = w_t - \eta \frac{\tilde{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$

**Pros**: Faster convergence than Adam on some problems, combines Nesterov correction with adaptive LR.
**Cons**: More hyperparameters, marginal improvement over Adam in practice.

## Convergence Comparison

| Optimizer | Convergence speed | Generalization | Memory | Tuning difficulty |
|---|---|---|---|---|
| SGD | Slow | Best | $O(n)$ | Hard (LR + schedule) |
| SGD + Momentum | Medium | Excellent | $O(2n)$ | Medium |
| NAG | Medium-Fast | Excellent | $O(2n)$ | Medium |
| AdaGrad | Fast (sparse) | Poor late | $O(2n)$ | Easy |
| RMSProp | Fast | Good | $O(2n)$ | Medium |
| Adam | Fast | Good | $O(3n)$ | Easy |
| AdamW | Fast | Excellent | $O(3n)$ | Easy |
| Nadam | Fast | Good | $O(3n)$ | Easy |

$n$ = number of parameters.

## Practical Recommendations

- **Start with AdamW** ($\eta = 3 \times 10^{-4}$, weight decay $= 0.01$) for quick results.
- **Switch to SGD + Momentum** ($\eta = 0.01$–$0.1$, momentum $=0.9$, with cosine annealing) if you want best generalization.
- **Use Adam** for Transformers and NLP tasks (with learning rate warmup).
- **Use RMSProp** or **Adam** for RNNs and reinforcement learning.
- **Use AdaGrad** for extremely sparse features (embedding layers with large vocabularies).
- **Prefer AdamW over Adam** — the decoupled weight decay almost always helps.
