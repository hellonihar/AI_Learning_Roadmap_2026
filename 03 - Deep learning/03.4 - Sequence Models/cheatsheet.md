# Sequence Models Cheatsheet

## Vanilla RNN

### Forward Pass
| Equation | Description |
|----------|-------------|
| $\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$ | Hidden state update |
| $\hat{\mathbf{y}}_t = \text{softmax}(\mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y)$ | Output (classification) |
| $\hat{\mathbf{y}}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y$ | Output (regression) |

### BPTT
| Equation | Description |
|----------|-------------|
| $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{hh}}$ | Accumulate gradients across time |
| $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \text{diag}(1 - \mathbf{h}_t^2) \, \mathbf{W}_{hh}$ | Jacobian of hidden state |

### Tensor Shapes
| Tensor | Shape |
|--------|-------|
| $\mathbf{x}_t$ | $(d_x, 1)$ |
| $\mathbf{h}_t$ | $(d_h, 1)$ |
| $\mathbf{W}_{xh}$ | $(d_h, d_x)$ |
| $\mathbf{W}_{hh}$ | $(d_h, d_h)$ |
| $\hat{\mathbf{y}}_t$ | $(d_y, 1)$ |

---

## LSTM

### Gate Equations
| Gate | Equation | Purpose |
|------|----------|---------|
| Forget | $\mathbf{f}_t = \sigma(\mathbf{W}_{hf} \mathbf{h}_{t-1} + \mathbf{W}_{xf} \mathbf{x}_t + \mathbf{b}_f)$ | What to discard from cell state |
| Input | $\mathbf{i}_t = \sigma(\mathbf{W}_{hi} \mathbf{h}_{t-1} + \mathbf{W}_{xi} \mathbf{x}_t + \mathbf{b}_i)$ | What new info to write |
| Candidate | $\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_{hc} \mathbf{h}_{t-1} + \mathbf{W}_{xc} \mathbf{x}_t + \mathbf{b}_c)$ | Candidate cell values |
| Output | $\mathbf{o}_t = \sigma(\mathbf{W}_{ho} \mathbf{h}_{t-1} + \mathbf{W}_{xo} \mathbf{x}_t + \mathbf{b}_o)$ | What to output |

### State Updates
| Update | Equation |
|--------|----------|
| Cell state | $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$ |
| Hidden state | $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$ |

### Gradient Flow
| Property | Value |
|----------|-------|
| $\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}}$ | $\mathbf{f}_t$ (element-wise) |
| CEC | Constant Error Carousel: error flows via $\mathbf{c}_t$ with minimal attenuation |

### Peephole Connections
| Gate with Peephole |
|--------------------|
| $\mathbf{f}_t = \sigma(\mathbf{W}_{hf} \mathbf{h}_{t-1} + \mathbf{W}_{xf} \mathbf{x}_t + \mathbf{W}_{cf} \mathbf{c}_{t-1} + \mathbf{b}_f)$ |
| $\mathbf{i}_t = \sigma(\mathbf{W}_{hi} \mathbf{h}_{t-1} + \mathbf{W}_{xi} \mathbf{x}_t + \mathbf{W}_{ci} \mathbf{c}_{t-1} + \mathbf{b}_i)$ |
| $\mathbf{o}_t = \sigma(\mathbf{W}_{ho} \mathbf{h}_{t-1} + \mathbf{W}_{xo} \mathbf{x}_t + \mathbf{W}_{co} \mathbf{c}_t + \mathbf{b}_o)$ |

### Tensor Shapes (LSTM)
| Tensor | Shape | Count |
|--------|-------|-------|
| Gate weight matrices | $(d_h, d_x + d_h)$ | 4 |
| Gate biases | $(d_h, 1)$ | 4 |
| $\mathbf{c}_t$ | $(d_h, 1)$ | 1 |

---

## GRU

### Gate Equations
| Gate | Equation | Purpose |
|------|----------|---------|
| Reset | $\mathbf{r}_t = \sigma(\mathbf{W}_{hr} \mathbf{h}_{t-1} + \mathbf{W}_{xr} \mathbf{x}_t + \mathbf{b}_r)$ | How much past to forget |
| Update | $\mathbf{z}_t = \sigma(\mathbf{W}_{hz} \mathbf{h}_{t-1} + \mathbf{W}_{xz} \mathbf{x}_t + \mathbf{b}_z)$ | Balance old vs new |

### State Updates
| Update | Equation |
|--------|----------|
| Candidate | $\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_{hh} (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$ |
| Final hidden | $\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t$ |

---

## Bidirectional RNN

$$
\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t ; \overleftarrow{\mathbf{h}}_t]
$$

- Forward: $\overrightarrow{\mathbf{h}}_t = \text{RNN}(\mathbf{x}_t, \overrightarrow{\mathbf{h}}_{t-1})$
- Backward: $\overleftarrow{\mathbf{h}}_t = \text{RNN}(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1})$

## Parameter Counts

| Model | Parameters |
|-------|-----------|
| Vanilla RNN | $d_h^2 + d_h d_x + d_h + d_y d_h + d_y$ |
| LSTM | $4(d_h^2 + d_h d_x + d_h) + d_y d_h + d_y$ |
| GRU | $3(d_h^2 + d_h d_x + d_h) + d_y d_h + d_y$ |

---

## Common Activation Functions

| Function | Range | Derivative |
|----------|-------|------------|
| $\sigma(x) = \frac{1}{1 + e^{-x}}$ | $(0, 1)$ | $\sigma(x)(1 - \sigma(x))$ |
| $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, 1)$ | $1 - \tanh^2(x)$ |

---

## Training Tips

| Technique | Purpose |
|-----------|---------|
| Gradient clipping | Prevent exploding gradients |
| Truncated BPTT | Limit unroll length for efficiency |
| Layer normalization | Stabilize hidden state distributions |
| Variational dropout | Regularize across time steps |
