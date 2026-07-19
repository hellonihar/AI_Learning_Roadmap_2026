# Building Models with `nn.Module`

`torch.nn.Module` is the base class for all neural network components. It provides parameter registration, train/eval mode switching, device movement, and serialization.

## The Basics

```python
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)   # 10 input features → 1 output

    def forward(self, x):
        return self.linear(x)
```

Every subclass must:
1. Call `super().__init__()` in `__init__`.
2. Define layers as attributes.
3. Implement `forward(self, x)` — the forward pass.

## Available Layers

```python
nn.Linear(in_features, out_features, bias=True)
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
nn.BatchNorm1d(num_features)
nn.Dropout(p=0.5)
nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
nn.Embedding(num_embeddings, embedding_dim)
nn.Flatten()
```

## nn.Sequential

For simple feed-forward architectures without custom logic:

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

Access sub-modules by index: `model[0]` returns the first `Linear`.

## Custom Modules with Multiple Heads

```python
class TwoTowerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.tower_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.tower_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x_a, x_b):
        out_a = self.tower_a(x_a)
        out_b = self.tower_b(x_b)
        combined = torch.cat([out_a, out_b], dim=-1)
        return self.classifier(combined)
```

## The `forward` Method

`forward` defines the computation. It can contain arbitrary Python logic — loops, conditionals, reshaping — and still be differentiable.

```python
class FlexibleMLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

**Never call `forward` directly — use `model(x)`.** The `__call__` method hooks in pre/post-forward processing (hooks, train/eval mode) that `forward` bypasses.

## Parameters & named_parameters

`nn.Module` automatically registers any `nn.Parameter` assigned as an attribute.

```python
class MyLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return x @ self.W + self.b

model = MyLayer(5, 2)
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
# W: torch.Size([5, 2])
# b: torch.Size([2])
```

**`parameters()`** returns all parameters as an iterable (for optimizers). **`named_parameters()`** also yields their names — useful for weight decay exclusion, freezing, or logging.

### Freezing Parameters

```python
for param in model.parameters():
    param.requires_grad = False

# Or selectively
for name, param in model.named_parameters():
    if "bias" in name:
        param.requires_grad = False
```

## Sub-Module Registration

Any `nn.Module` assigned as an attribute is automatically registered as a sub-module. Parameters nested inside sub-modules appear in `parameters()` and `named_parameters()`.

```python
class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(...)   # registered
        self.classifier = nn.Linear(...)     # registered
        self.custom_param = nn.Parameter(...) # registered

        # NOT registered — won't be seen by optimizers or .to(device)
        self.not_a_param = torch.randn(10, 10)
```

Put tensors that are not parameters in a plain list or use `register_buffer()` for non-learnable persistent state.

## Module Utilities

```python
model = MyModel()

model.to("cuda")                # moves all parameters to GPU
model.train()                   # enables dropout, batch norm stats
model.eval()                    # disables dropout, freezes BN stats
model.state_dict()              # dict of all parameters (for saving)
model.load_state_dict(state)    # load from saved state_dict
model.apply(init_weights)       # apply function to every sub-module
model.zero_grad()               # zero all parameter gradients
model(x)                        # forward pass (calls __call__ → forward)
```

> **Design guideline:** Keep modules small and composable. A well-designed module does one thing. Compose them hierarchically for complex architectures.
