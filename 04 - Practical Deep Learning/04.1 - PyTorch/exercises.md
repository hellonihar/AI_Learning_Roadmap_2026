# Exercises

## Exercise 1: Custom Dataset

Write a `CSVDataset` class that loads data from a CSV file path given at init. Assume the last column is the target (float for regression). Include `__len__` and `__getitem__`. Add an optional `transform` parameter.

<details>
<summary>Answer</summary>

```python
import torch
import pandas as pd
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path)
        self.data = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
        self.targets = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
```
</details>

---

## Exercise 2: 3-Layer MLP

Define a 3-layer MLP using `nn.Module` with ReLU activations after each hidden layer. The architecture: 784 → 512 → 256 → 10. Do **not** use `nn.Sequential`. Then, adapt it to use `nn.Sequential`.

<details>
<summary>Answer</summary>

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

# Sequential version
MLP_seq = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
```
</details>

---

## Exercise 3: Manual Autograd (No nn.Module)

Train a linear regression model `y = x @ w + b` on synthetic data using only `torch` tensors and manual gradient descent. Do not use `nn.Module` or `optim`. Use `requires_grad` and `backward()`.

<details>
<summary>Answer</summary>

```python
import torch

# Synthetic data
torch.manual_seed(42)
X = torch.randn(100, 3)
true_w = torch.tensor([[2.0], [-1.0], [0.5]])
true_b = torch.tensor([0.1])
y = X @ true_w + true_b + 0.05 * torch.randn(100, 1)

# Initialize parameters
w = torch.randn(3, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.01
n_epochs = 500

for epoch in range(n_epochs):
    y_pred = X @ w + b
    loss = ((y_pred - y) ** 2).mean()

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}: loss = {loss.item():.6f}")

print(f"w = {w.squeeze().detach().numpy()}")
print(f"b = {b.item():.4f}")
```
</details>

---

## Exercise 4: Gradient Accumulation

Take a standard training loop that uses batch_size=128. Modify it to simulate batch_size=512 using gradient accumulation over 4 micro-batches. Show the key part of the loop.

<details>
<summary>Answer</summary>

```python
accumulation_steps = 4  # effective batch = 128 * 4 = 512

model.train()
optimizer.zero_grad()  # zero once before accumulation

for batch_idx, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device), targets.to(device)

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss = loss / accumulation_steps    # normalize
    loss.backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()           # zero after step

# Handle leftover batches (if drop_last=False)
if (batch_idx + 1) % accumulation_steps != 0:
    optimizer.step()
    optimizer.zero_grad()
```
</details>

---

## Exercise 5: Optimizer Comparison

Train a small MLP on the synthetic dataset from the code walkthrough using three different optimizers (SGD, Adam, AdamW). Compare the validation loss curves after 10 epochs. Write a script that runs all three and produces a table.

<details>
<summary>Answer</summary>

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Reuse SyntheticDataset and MLP from code_walkthrough.md
from code_walkthrough import SyntheticDataset, MLP  # assuming in same dir

def train_with_optimizer(opt_class, opt_kwargs, name):
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = opt_class(model.parameters(), **opt_kwargs)

    model.train()
    for epoch in range(10):
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(train_loader)

dataset = SyntheticDataset()
train_ds, val_ds = random_split(dataset, [8000, 2000])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = {}
for name, cls, kwargs in [
    ("SGD",   optim.SGD,   {"lr": 0.01, "momentum": 0.9}),
    ("Adam",  optim.Adam,  {"lr": 0.001}),
    ("AdamW", optim.AdamW, {"lr": 0.001, "weight_decay": 0.01}),
]:
    loss = train_with_optimizer(cls, kwargs, name)
    results[name] = loss

print(f"{'Optimizer':<10} {'Final Loss':<12}")
print("-" * 22)
for name, loss in results.items():
    print(f"{name:<10} {loss:.4f}")
```
</details>

---

## Exercise 6: Freeze & Fine-tune

Given a pretrained model with `features` and `classifier` sub-modules, freeze all parameters in `features` and train only `classifier` for 5 epochs.

<details>
<summary>Answer</summary>

```python
# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# Only classifier params will be updated
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```
</details>

---

## Exercise 7: Custom Loss with Autograd

Implement the **Huber loss** manually using only `torch` operations and autograd:

```
L(y, y_hat) = 0.5 * (y - y_hat)^2          if |y - y_hat| <= delta
              delta * |y - y_hat| - 0.5 * delta^2   otherwise
```

Verify it produces the same result as `torch.nn.HuberLoss(delta=1.0)`.

<details>
<summary>Answer</summary>

```python
def huber_loss(y_pred, y_true, delta=1.0):
    diff = y_pred - y_true
    is_small = diff.abs() <= delta
    squared = 0.5 * diff ** 2
    linear = delta * diff.abs() - 0.5 * delta ** 2
    return torch.where(is_small, squared, linear).mean()

# Verification
y_pred = torch.randn(32, 10, requires_grad=True)
y_true = torch.randn(32, 10)

my_loss = huber_loss(y_pred, y_true)
ref_loss = nn.HuberLoss(delta=1.0)(y_pred, y_true)

print(f"Difference: {(my_loss - ref_loss).item():.2e}")  # ~0.0

my_loss.backward()
print("Custom Huber grad:", y_pred.grad.norm().item())
```
</details>

---

## Exercise 8: Save/Load with Metadata

Extend the training loop from the code walkthrough to save a checkpoint every 5 epochs. Each checkpoint should include: epoch number, model state dict, optimizer state dict, scheduler state dict, and the current learning rate. Then write a function that loads the best checkpoint (by validation accuracy).

<details>
<summary>Answer</summary>

```python
def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_acc": val_acc,
        "lr": optimizer.param_groups[0]["lr"],
    }, path)

def load_best_checkpoint(model, optimizer, scheduler, path):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"Loaded epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.2f}%, lr={ckpt['lr']:.2e})")
    return ckpt["epoch"]

# Usage in loop
best_acc = 0.0
for epoch in range(30):
    train_one_epoch()
    val_acc = validate()

    if (epoch + 1) % 5 == 0:
        save_checkpoint(model, optimizer, scheduler, epoch + 1, val_acc,
                        f"checkpoint_epoch_{epoch+1:03d}.pt")

    if val_acc > best_acc:
        best_acc = val_acc
        save_checkpoint(model, optimizer, scheduler, epoch + 1, val_acc,
                        "best_model.pt")

# Load best
load_best_checkpoint(model, optimizer, scheduler, "best_model.pt")
```
</details>
