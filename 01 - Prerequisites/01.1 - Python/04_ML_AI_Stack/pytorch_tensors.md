# PyTorch: Tensors, Autograd, nn.Module

## Tensors
```python
import torch

x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
x = torch.randn(32, 3, 224, 224)  # batch images
x = torch.zeros(10, 20)
x = torch.ones(10, 20)
x = torch.arange(100)
x.device  # cpu / cuda
x = x.cuda() if torch.cuda.is_available() else x
```

## Operations
```python
x + y; x - y; x * y; x / y
x @ y.T           # matrix multiply
torch.cat([x, y], dim=0)
x.view(5, -1)     # reshape
x.permute(0, 2, 1) # transpose dims
```

## Autograd
```python
x = torch.randn(3, requires_grad=True)
y = (x**2).sum()
y.backward()       # compute gradients
x.grad             # d(y)/d(x)
```

## nn.Module
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10)
)

# Custom module
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

## Training Loop
```python
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for X_batch, y_batch in dataloader:
        optim.zero_grad()
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optim.step()
```
