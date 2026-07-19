# Code Walkthrough: Full PyTorch Training Pipeline

A complete end-to-end example training a 3-layer MLP on synthetic data with mixed precision, checkpointing, and evaluation.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

## 1. Synthetic Dataset

```python
class SyntheticDataset(Dataset):
    def __init__(self, n_samples=10000, n_features=20, n_classes=10, seed=42):
        rng = torch.Generator().manual_seed(seed)
        self.data = torch.randn(n_samples, n_features, generator=rng)
        # Create a simple linear decision boundary + noise
        w = torch.randn(n_features, n_classes, generator=rng)
        logits = self.data @ w
        self.targets = logits.argmax(dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
```

## 2. Model Definition

```python
class MLP(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
```

## 3. Data Preparation

```python
full_dataset = SyntheticDataset(n_samples=10000)
train_len = int(0.8 * len(full_dataset))
val_len = len(full_dataset) - train_len
train_ds, val_ds = random_split(full_dataset, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, pin_memory=True)
```

## 4. Training Setup

```python
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# Mixed precision
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

num_epochs = 20
best_val_acc = 0.0
```

## 5. Training Loop

```python
for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()

    train_loss /= train_total
    train_acc = 100.0 * train_correct / train_total

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    val_loss /= val_total
    val_acc = 100.0 * val_correct / val_total

    scheduler.step()

    print(f"Epoch {epoch+1:2d}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    # --- Checkpoint best model ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
        }, "best_checkpoint.pt")
        print(f"  → New best model saved (val_acc={val_acc:.2f}%)")
```

## 6. Evaluation

```python
# Load best checkpoint
checkpoint = torch.load("best_checkpoint.pt", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
      f"with val_acc={checkpoint['val_acc']:.2f}%")

# Final evaluation
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().tolist())
        all_targets.extend(targets.tolist())

accuracy = 100.0 * sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
print(f"Final validation accuracy: {accuracy:.2f}%")
```

## 7. Inference on New Data

```python
model.eval()
x_new = torch.randn(5, 20, device=device)

with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
    logits = model(x_new)
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)

print("Predictions:", preds.cpu().tolist())
print("Probabilities:", probs.cpu().numpy().round(3))
```

## Expected Output

```
Using device: cuda
Epoch  1/20 | Train Loss: 1.8612 Acc: 43.21% | Val Loss: 1.5820 Acc: 55.10%
  → New best model saved (val_acc=55.10%)
Epoch  2/20 | Train Loss: 1.3498 Acc: 63.84% | Val Loss: 1.2194 Acc: 68.35%
  → New best model saved (val_acc=68.35%)
...
Epoch 20/20 | Train Loss: 0.0142 Acc: 99.86% | Val Loss: 0.0559 Acc: 98.70%
  → New best model saved (val_acc=98.70%)

Final validation accuracy: 98.70%
```

This walkthrough demonstrates every component working together: datasets, model definition, mixed precision training, checkpointing, and final evaluation. Adapt the model architecture, dataset, and hyperparameters for your own tasks.
