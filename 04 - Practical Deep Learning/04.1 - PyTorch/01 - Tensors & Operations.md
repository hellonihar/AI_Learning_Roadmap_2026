# Tensors & Operations

PyTorch tensors are the fundamental data structure — comparable to NumPy arrays but with GPU acceleration and automatic differentiation built in. Nearly everything in PyTorch begins with a tensor.

## Creating Tensors

```python
import torch
import numpy as np

# From a Python list
t1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# From a NumPy array (shares memory by default)
arr = np.array([1.0, 2.0, 3.0])
t2 = torch.from_numpy(arr)

# Factory functions
zeros = torch.zeros(3, 4)
ones  = torch.ones(2, 3)
eye   = torch.eye(5)
rand  = torch.randn(3, 3)          # standard normal
randu = torch.rand(3, 3)           # uniform [0, 1)
arange = torch.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5) # [0.0, 0.25, 0.5, 0.75, 1.0]
```

## dtype & Device

Every tensor has a data type and lives on a device.

```python
t = torch.tensor([1, 2, 3], dtype=torch.float32)
print(t.dtype, t.device)   # torch.float32, cpu

# Move to GPU (if available)
if torch.cuda.is_available():
    t = t.to("cuda")
    t = t.cuda()            # equivalent shorthand

# Common dtypes: float32 (default), float64, int64, bool, bfloat16
half = t.half()             # float16
double = t.double()         # float64
```

**Device semantics:** operations between tensors require both to be on the same device. Mixing CPU and GPU tensors raises a runtime error.

## Shape Manipulation

```python
x = torch.randn(2, 3, 4)

# Reshape (total elements must match)
x_flat = x.reshape(-1)              # 1-D, 24 elements
x_2x12 = x.reshape(2, 12)

# View — like reshape but must be contiguous
x_view = x.view(2, 12)

# Squeeze — remove dimensions of size 1
y = torch.randn(1, 3, 1, 4)
y_sq = y.squeeze()                  # (3, 4)
y_sq_dim0 = y.squeeze(0)            # (3, 1, 4)

# Unsqueeze — insert a dimension of size 1
z = torch.randn(3, 4)
z_3d = z.unsqueeze(0)               # (1, 3, 4)
z_3d_last = z.unsqueeze(-1)         # (3, 4, 1)

# Transpose
x_t = x.transpose(0, 1)             # swap dim 0 and 1 → (3, 2, 4)
x_perm = x.permute(2, 0, 1)        # arbitrary reorder → (4, 2, 3)
```

**`reshape` vs `view`:** `view` requires the tensor to be contiguous in memory (use `.contiguous()` first if unsure). `reshape` returns a view when possible, otherwise a copy — safer but slightly less predictable.

## Basic Operations

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Arithmetic
c = a + b               # [5, 7, 9]
c = a - b               # [-3, -3, -3]
c = a * b               # element-wise: [4, 10, 18]
c = a / b
c = a ** 2

# In-place (ends with _) — use sparingly, breaks autograd history
a.add_(b)

# Matrix multiplication
m1 = torch.randn(3, 4)
m2 = torch.randn(4, 5)
mm = m1 @ m2                     # (3, 5) — @ operator
mm = torch.matmul(m1, m2)        # same
mm = m1.mm(m2)                   # 2-D only

# Batched matmul
b1 = torch.randn(10, 3, 4)
b2 = torch.randn(10, 4, 5)
out = torch.bmm(b1, b2)          # (10, 3, 5)
```

## Broadcasting

PyTorch follows NumPy broadcasting rules: two tensors are compatible when their dimensions are equal or one of them is 1 (or missing).

```python
a = torch.randn(3, 1)       # (3, 1)
b = torch.randn(5)          # (5,)       → treated as (1, 5)
c = a + b                   # (3, 5) — broadcast both

# Common use: adding bias
x = torch.randn(32, 128)    # batch of 32, 128 features
bias = torch.randn(128)     # (128,) → (1, 128) → broadcast to (32, 128)
out = x + bias
```

## Reduction & Indexing

```python
x = torch.randn(3, 4)

x.sum()                     # scalar
x.mean(dim=0)               # mean over batch → (4,)
x.max(dim=1)                # returns (values, indices)

# NumPy-like indexing
x[0, :]                     # first row
x[:, -1]                    # last column
x[x > 0]                    # boolean masking → 1-D tensor of positive values
```

## When to Use What

| Task | Function |
|------|----------|
| Create from data | `torch.tensor()`, `torch.from_numpy()` |
| Fixed-value tensors | `zeros`, `ones`, `eye`, `full` |
| Random | `randn`, `rand`, `randint`, `randperm` |
| Change shape | `reshape`, `view`, `squeeze`, `unsqueeze`, `permute` |
| Math | `+ - * / @`, `torch.matmul`, `torch.bmm` |
| Move device | `.to()`, `.cuda()`, `.cpu()` |
| Change type | `.float()`, `.half()`, `.long()`, `.bool()` |

> **Key insight:** Every tensor operation produces a new tensor (unless in-place). This functional style is what allows PyTorch's autograd to track the computational graph.
