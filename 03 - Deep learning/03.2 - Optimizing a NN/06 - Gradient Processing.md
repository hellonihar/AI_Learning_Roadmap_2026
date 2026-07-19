# Gradient Processing

Gradient processing techniques modify how gradients are computed or applied to stabilize training and fit models into limited memory.

## Gradient Clipping

Gradient clipping prevents exploding gradients by capping gradient magnitude before the update step.

### Norm Clipping

Scale down the entire gradient vector if its $L_2$ norm exceeds a threshold $c$:

$$
g \leftarrow \begin{cases}
g & \text{if } \|g\|_2 \leq c \\
c \cdot \frac{g}{\|g\|_2} & \text{if } \|g\|_2 > c
\end{cases}
$$

**Intuition**: Preserves direction while limiting step size. The update direction remains correct, but the magnitude is bounded.

**When to use**: RNNs/LSTMs (where exploding gradients are common). Large Transformer training. Typically $c \in [0.5, 10]$.

### Value Clipping

Clip each element individually to $[-\text{clip}, \text{clip}]$:

$$ g_i \leftarrow \max(\min(g_i, \text{clip}), -\text{clip}) $$

Coarser than norm clipping — distorts direction. Rarely used in practice (norm clipping is preferred).

## Gradient Accumulation

Simulates a larger batch size by accumulating gradients over multiple mini-batches before updating:

```python
optimizer.zero_grad()
for micro_step in range(accumulation_steps):
    loss = model(x[micro_step * micro_batch : (micro_step + 1) * micro_batch])
    loss = loss / accumulation_steps  # scale loss
    loss.backward()
optimizer.step()
```

**Intuition**: The sum of gradients over $N$ micro-batches of size $B$ equals the gradient of one batch of size $N \cdot B$ (up to BatchNorm differences). Each micro-batch's gradients are accumulated in-place.

**When to use**: Limited GPU memory. Train with effective batch size 256 on a GPU that only fits batch size 32.

**Tradeoffs**: Increases training time linearly with accumulation steps (forward/backward $N$ times per update). May affect BatchNorm statistics (use sync BatchNorm if available).

## Gradient Checkpointing

Trades compute for memory by not storing intermediate activations for all layers. During the forward pass, only "checkpoint" layers save activations. During backward pass, missing activations are recomputed from the nearest checkpoint.

```python
# Standard: stores all activations O(L * memory_per_layer)
memory = sum(layer_memory for layer in layers)

# Checkpointing: stores activations every sqrt(L) layers
memory = sqrt(L) * memory_per_layer
compute = 1.5 * standard_forward_compute  # ~1 extra forward pass
```

**Intuition**: Recomputing activations during backward pass avoids storing them all simultaneously. The memory savings are $O(L)$ → $O(\sqrt{L})$ with $\sqrt{L}$ checkpoints.

**When to use**: Very deep networks (ResNet-152, GPT, ViT-Large). Training on single GPU with limited VRAM.

**Tradeoffs**: Increases computation ~15–33% (extra forward pass). Throughput decrease depends on model size and checkpoint frequency.

## Summary

| Technique | Solves | Cost |
|---|---|---|
| Gradient clipping (norm) | Exploding gradients | Negligible |
| Gradient accumulation | Batch size too large for memory | Linear time increase |
| Gradient checkpointing | Activations exceed memory | 15–33% more compute |
