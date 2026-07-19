# Regularization

Overfitting occurs when a model learns the training data too well, including its noise, and fails to generalize to unseen data. Regularization encompasses techniques that penalize model complexity or inject noise during training to force the network to learn simpler, more robust patterns.

## L1 and L2 Regularization (Weight Decay)

The most common form of regularization adds a penalty term to the loss function proportional to the magnitude of the weights.

### L2 Regularization (Ridge)

$$ \mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{data}} + \frac{\lambda}{2} \sum_{i} w_i^2 $$

The gradient of the penalty term is $\lambda w$, so the weight update becomes:

$$ w \leftarrow w - \eta (\nabla \mathcal{L}_{\text{data}} + \lambda w) = w(1 - \eta \lambda) - \eta \nabla \mathcal{L}_{\text{data}} $$

The $(1 - \eta \lambda)$ term decays the weight multiplicatively — hence the name **weight decay**. L2 regularization shrinks weights proportionally to their size, encouraging the network to use all weights a little rather than a few weights a lot. This prevents any single neuron from dominating and forces distributed representations.

**Intuition**: A large weight means the model is highly sensitive to a specific feature. If that feature is noisy, the model overfits. L2 penalizes large weights, so the network learns smoother decision boundaries that generalize better.

### L1 Regularization (Lasso)

$$ \mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{data}} + \lambda \sum_{i} |w_i| $$

The gradient is $\lambda \cdot \text{sign}(w)$, pushing weights toward zero with a constant force regardless of magnitude. This produces **sparse** weight vectors — many weights become exactly zero.

**When to use**: L1 when you want feature selection (sparse models). L2 when you want all features used modestly. L1 + L2 (Elastic Net) combines both.

## Dropout

Introduced by Srivastava et al. (2014), dropout randomly "drops" neurons during training by setting their output to zero with probability $p$ (a hyperparameter, typically 0.2–0.5).

During each forward pass, a binary mask $m \sim \text{Bernoulli}(1-p)$ is sampled and applied:

$$ \hat{h} = m \odot h $$

During inference, all neurons are active but scaled by $(1-p)$ to preserve expected output magnitude:

$$ h_{\text{inference}} = (1-p) \cdot h $$

Or equivalently, during training, activations are scaled by $1/(1-p)$ so no scaling is needed at test time (inverted dropout, used in practice).

**Intuition**: Dropout prevents co-adaptation of neurons. Each neuron must learn useful features independently, since it cannot rely on a fixed set of partners being present. This is analogous to training an ensemble of $2^n$ sub-networks (where $n$ is the number of neurons) and averaging them at test time. The result is a robust, generalizing model.

**Practical tips**: Use $p=0.5$ for fully connected layers, $p=0.2$–$0.3$ for recurrent layers, and little to no dropout in convolutional layers (use Batch Norm instead).

## Data Augmentation

Data augmentation artificially expands the training set by applying label-preserving transformations to input data. It is the most effective regularization technique for vision, text, and audio.

**Vision**: Random crops, horizontal flips, rotations, color jitter, cutout, MixUp, CutMix.
**Text**: Synonym replacement, back-translation, random insertion/deletion.
**Audio**: Time stretching, pitch shifting, adding background noise.

**Intuition**: By training on transformed versions of inputs, the network learns invariance to those transformations. It cannot memorize pixel-perfect patterns because the same object appears differently each epoch. This forces the model to learn the true underlying concept rather than surface statistics.

**Why it works**: Data augmentation increases effective dataset size without collecting new data. More data = less overfitting. It encodes inductive biases directly into the training distribution.

## Early Stopping

Early stopping monitors validation loss during training and halts when it stops improving. A copy of the model parameters at the best validation step is saved.

**Algorithm**: Train for $N$ epochs. After each epoch, compute validation loss $\mathcal{L}_{\text{val}}$. If $\mathcal{L}_{\text{val}}$ has not improved for $p$ epochs (patience), restore the best parameters and stop.

**Intuition**: As training progresses, the model first captures the dominant, generalizable patterns (low-bias regime). Eventually it begins fitting noise specific to the training set. Validation loss initially decreases, reaches a minimum, then rises. Early stopping picks the iteration right before overfitting begins.

**Connection to L2**: Early stopping is equivalent to L2 regularization with $\lambda = 1/(2\eta t_{\text{stop}})$ under certain conditions, where $t_{\text{stop}}$ is the stopping iteration.

| Technique | How it reduces overfitting | Computational cost |
|---|---|---|
| L2 | Shrinks weights proportionally | Minimal (gradient addition) |
| L1 | Zeroes out unimportant weights | Minimal |
| Dropout | Trains ensemble of subnetworks | Forward/backward through mask (no extra params) |
| Data Augmentation | Increases effective dataset size | Forward pass on augmented data |
| Early Stopping | Limits model capacity implicitly | Free (monitors validation loss) |

## Practical Workflow

1. Always use **data augmentation** first — it's free accuracy.
2. Add **L2 regularization** ($\lambda \approx 10^{-4}$ to $10^{-3}$) as a baseline.
3. If overfitting persists, add **Dropout** before fully connected layers.
4. Use **early stopping** as a safety net.
5. For sparse models, substitute L1 or Elastic Net for L2.
