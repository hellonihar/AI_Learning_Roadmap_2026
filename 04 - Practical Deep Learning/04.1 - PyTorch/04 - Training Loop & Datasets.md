# Training Loop & Datasets

A training loop is where everything comes together: data loading, forward pass, loss computation, backpropagation, and parameter updates. PyTorch provides `Dataset` and `DataLoader` to handle data efficiently.

## Dataset & DataLoader

`torch.utils.data.Dataset` is an abstract class for representing a dataset. You override `__len__` and `__getitem__`.

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
```

`DataLoader` wraps a `Dataset` and provides batching, shuffling, and parallel loading.

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,        # parallel data loading
    pin_memory=True,      # faster GPU transfer
    drop_last=True,       # drop incomplete final batch
)
```

## Train / Val Split

Split your dataset before wrapping in `DataLoader`:

```python
from torch.utils.data import random_split

dataset = MyDataset(full_data, full_targets)
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
```

## Standard Training Loop

```python
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()                          # enable training mode
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # 1. Zero gradients
        optimizer.zero_grad()

        # 2. Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 3. Backward pass
        loss.backward()

        # 4. Update parameters
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # --- Validation ---
    model.eval()                           # disable dropout, BN
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():                  # no gradient tracking
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
          f"val_loss={avg_val_loss:.4f}, acc={accuracy:.2f}%")
```

## The Four-Step Training Dance

Each batch follows the same pattern:

| Step | Code | What Happens |
|------|------|--------------|
| **Zero grad** | `optimizer.zero_grad()` | Clears old gradients from `.grad` |
| **Forward** | `outputs = model(inputs)` | Runs computation, builds graph |
| **Backward** | `loss.backward()` | Computes gradients via chain rule |
| **Step** | `optimizer.step()` | Updates parameters using gradients |

**Why `zero_grad()`?** Gradients accumulate by default. Without resetting, each backward call adds to existing gradients.

## model.train() vs model.eval()

These methods set the module and all its children to the specified mode.

| Feature | `train()` | `eval()` |
|---------|-----------|----------|
| Dropout | Active (randomly drops) | Disabled (scales activations) |
| BatchNorm | Uses batch stats | Uses running averages |
| requires_grad | Unchanged | Unchanged (use `no_grad` separately) |

**Always call `model.eval()` before validation/testing** and pair it with `torch.no_grad()` for maximum efficiency.

## Advanced: Logging & Early Stopping

```python
best_val_loss = float("inf")
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    train_one_epoch(...)
    val_loss = validate(...)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
```

## Common Pitfalls

1. **Forgetting `.to(device)`:** Data stays on CPU while model is on GPU — silent error or crash.
2. **Skipping `zero_grad()`:** Gradients accumulate across batches, causing incorrect updates.
3. **Calling `model.eval()` before training:** Disables dropout — model underfits.
4. **`loss.item()` after `backward()`:** Fine, but `loss` is still part of the graph — use `loss.detach().item()` or just `loss.item()` (works for scalars).
5. **Shuffling validation:** Wasteful — validation metrics are identical regardless of order.

> **Key takeaway:** Master this four-step loop. It appears in every PyTorch training script, from simple MLPs to large-scale distributed training.
