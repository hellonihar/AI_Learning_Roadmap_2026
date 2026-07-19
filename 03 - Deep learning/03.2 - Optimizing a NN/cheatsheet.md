# Cheatsheet

## Optimizer Update Rules

| Optimizer | Update Rule |
|---|---|
| SGD | $w_{t+1} = w_t - \eta g_t$ |
| Momentum | $v_{t+1} = \beta v_t + g_t$, $w_{t+1} = w_t - \eta v_{t+1}$ |
| NAG | $v_{t+1} = \beta v_t + \nabla \mathcal{L}(w_t - \eta \beta v_t)$, $w_{t+1} = w_t - \eta v_{t+1}$ |
| AdaGrad | $G_t = G_{t-1} + g_t^2$, $w_{t+1} = w_t - \eta \frac{g_t}{\sqrt{G_t + \epsilon}}$ |
| RMSProp | $v_t = \beta v_{t-1} + (1-\beta) g_t^2$, $w_{t+1} = w_t - \eta \frac{g_t}{\sqrt{v_t + \epsilon}}$ |
| Adam | $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$, $w_{t+1} = w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$ |
| AdamW | Same as Adam but weight decay decoupled: $w_{t+1} = w_t - \eta(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda w_t)$ |
| Nadam | Adam with Nesterov momentum correction in $\hat{m}_t$ |

**Defaults**: SGD: $\eta = 0.01$–$0.1$, Momentum: $\beta = 0.9$, RMSProp: $\beta = 0.9$, Adam: $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$, AdamW: + weight decay $= 0.01$–$0.1$.

## Normalization Formulas

| Norm | Formula | Axes |
|---|---|---|
| BatchNorm | $\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$, $y_i = \gamma \hat{x}_i + \beta$ | $N, H, W$ |
| LayerNorm | $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$, $y_i = \gamma \hat{x}_i + \beta$ | $C, H, W$ |
| InstanceNorm | $\hat{x}_{nchw} = \frac{x_{nchw} - \mu_{nc}}{\sqrt{\sigma_{nc}^2 + \epsilon}}$ | $H, W$ |
| GroupNorm | $\hat{x}_i = \frac{x_i - \mu_{ng}}{\sqrt{\sigma_{ng}^2 + \epsilon}}$ | $(C/G), H, W$ |

## Regularization Summary

| Technique | Mechanism | Key Hyperparameter |
|---|---|---|
| L1 | Sparsity (zero weights) | $\lambda$ |
| L2 | Weight decay (small weights) | $\lambda$ |
| Dropout | Random neuron dropping | $p$ (drop probability) |
| Data Augmentation | Label-preserving transforms | Transforms, magnitude |
| Early Stopping | Stop at best val loss | Patience |

## Learning Rate Schedules

| Schedule | Formula | Best for |
|---|---|---|
| Step | $\eta_0 \cdot \gamma^{\lfloor t/k\rfloor}$ | CNNs |
| Exponential | $\eta_0 \cdot e^{-kt}$ | Smooth decay |
| Cosine | $\eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos(t\pi/T))$ | Modern baselines |
| Warmup | $\eta_0 \cdot \min(1, t/W)$ | Transformers, Adam |
| ReduceLROnPlateau | Reduce when metric plateaus | Auto-tuning |
| Cyclic | Cosine with periodic reset | Fast exploration |

## Gradient Processing

| Technique | Usage | Key Param |
|---|---|---|
| Norm clipping | $g \leftarrow c \cdot g / \|g\|_2$ if $\|g\|_2 > c$ | $c \in [0.5, 10]$ |
| Accumulation | Sum gradients over $N$ micro-batches | $N = \text{effective\_batch} / \text{micro\_batch}$ |
| Checkpointing | Recompute activations during backward | Checkpoint interval $\sqrt{L}$ |
