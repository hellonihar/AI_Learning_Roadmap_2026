# Autograd & Computational Graphs

Autograd is PyTorch's automatic differentiation engine. It records operations as you run them to build a computational graph, then traverses it backwards to compute gradients.

## requires_grad & Tracking

Every tensor has a `requires_grad` flag. When set to `True`, all operations on that tensor are tracked for later gradient computation.

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)
w = torch.tensor([1.5, -0.5], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)

# Forward computation
z = w @ x + b        # dot product + bias
loss = z.sum()       # scalar loss
```

After `loss.backward()`, gradients are accumulated in the `.grad` attribute of each leaf tensor.

```python
loss.backward()

print(w.grad)   # dy/dw = x → tensor([2.0, 3.0])
print(x.grad)   # None — x is not a leaf we want to optimize
print(b.grad)   # dy/db = 1 → tensor([1.0])
```

**Leaves** are tensors created directly by the user (not the result of operations). Only leaf tensors retain `.grad` — intermediate tensors have their gradients freed after `backward()` to save memory.

## Computational Graph Structure

When you run forward operations, PyTorch builds a **directed acyclic graph (DAG)**:

- **Nodes** = tensors and operations (functions)
- **Edges** = data flow between operations
- **Forward graph:** input → ops → output
- **Backward graph:** output → grad_fn chain → leaf `.grad`

```python
# Inspect the grad_fn
print(loss.grad_fn)          # <SumBackward0 at 0x...>
print(z.grad_fn)             # <AddBackward0 at 0x...>
print(w.grad_fn)             # None — leaf tensor
```

The backward graph is built lazily: `grad_fn` attributes form a linked list from output to inputs. Calling `backward()` traverses this chain, applying the chain rule at each node.

### Visualizing the Graph

```
x ─┐                   
    ├── dot ──→ z ──→ + ──→ loss ──→ backward()
w ─┘              ↑           ↑
                  b ──────────┘
```

Each operation has a corresponding `torch.autograd.Function` subclass (e.g., `AddBackward0`, `MmBackward0`, `SumBackward0`) that implements the backward pass.

## Gradient Accumulation

By default, `backward()` **accumulates** gradients into `.grad`. This is useful for:

1. **Simulating larger batch sizes** when GPU memory is limited.
2. **Gradient checkpointing** trade-offs.

```python
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()                    # accumulates into .grad

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()              # update weights
        optimizer.zero_grad()         # reset .grad to zero
```

**Warning:** If you don't call `zero_grad()`, gradients from every batch pile up. This is a common bug.

## Detaching Graphs

Sometimes you need to break the computational graph:

```python
x = torch.randn(3, requires_grad=True)
y = x * 2

z = y.detach()          # z has requires_grad=False, no graph connection
w = z * 3              # w won't propagate gradients to x

# .detach() vs .data (legacy, avoid .data)
y_detached = y.detach()
y_detached.zero_()      # modifies y too! Be careful.
```

Use `.detach()` when:
- Extracting a tensor for logging/plotting that shouldn't affect gradients.
- Implementing stop-gradient in GANs or contrastive learning.
- Freezing a feature extractor.

## Disabling Gradient Tracking

```python
# Option 1: no_grad context (recommended for inference)
with torch.no_grad():
    preds = model(x)        # no graph built, faster, less memory

# Option 2: set requires_grad to False
x.requires_grad_(False)

# Option 3: inference_mode (PyTorch 1.9+) — fastest
with torch.inference_mode():
    preds = model(x)
```

`torch.no_grad()` disables gradient computation entirely for the block. Use it in evaluation loops to reduce memory and speed up inference.

## What Autograd Tracks

| Operation | Tracked? | Notes |
|-----------|----------|-------|
| Tensor arithmetic (`+`, `-`, `*`, `/`) | Yes | |
| Matrix multiplication (`@`, `matmul`) | Yes | |
| Indexing & slicing | Yes | But no in-place on views |
| In-place ops (`add_`, `copy_`) | Yes | Can cause errors with shared memory |
| NumPy conversion | No | `.numpy()` requires detach |
| `.item()` | No | Returns Python scalar |
| Plain Python control flow | No | But graph still tracks through it |

## Common Pitfalls

1. **In-place modification after backward:** Modifying a tensor that requires grad after backward will raise an error — use `.data` or `.detach()` if you must.
2. **Non-leaf `.grad`:** Only leaves have `.grad` populated by default. Use `retain_grad()` on intermediates if needed (rare).
3. **Backward on non-scalar:** `backward()` requires a scalar loss. For non-scalar, pass a same-shape gradient tensor.

> **Summary:** Autograd builds a graph during forward, then traverses it backwards to compute gradients. Understanding the graph structure helps debug gradient flow issues and optimize memory usage.
