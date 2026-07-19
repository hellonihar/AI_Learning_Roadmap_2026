# PyTorch Cheatsheet

## Tensor Creation

```python
torch.tensor([1, 2, 3])                          # from list
torch.from_numpy(np_array)                       # from NumPy
torch.zeros(3, 4), torch.ones(2, 3)
torch.eye(5), torch.arange(0, 10, 2)
torch.randn(3, 3), torch.rand(3, 3)              # normal, uniform
torch.randint(0, 10, (5,)), torch.randperm(10)
torch.empty(3, 4), torch.full((2, 3), 7.0)
torch.linspace(0, 1, 5), torch.logspace(-3, 0, 10)
```

### dtype & Device

```python
t.float(), t.half(), t.long(), t.int(), t.bool()   # type casting
t.to(device), t.cuda(), t.cpu()                      # device transfer
t.dtype, t.device, t.shape, t.numel()                # introspection
```

### Shape Ops

```python
t.reshape(3, 4), t.view(3, 4)                       # reshape (view needs contiguous)
t.squeeze(), t.unsqueeze(0)                          # remove/add dim of size 1
t.transpose(0, 1), t.permute(2, 0, 1)               # dimension swap/reorder
t.expand(4, -1), t.repeat(2, 3)                      # broadcasting helpers
t.flatten(), t.flatten(start_dim=1)                   # flatten
t.contiguous()                                        # ensure contiguous memory
```

### Math

```python
a + b, a - b, a * b, a / b, a ** 2                  # element-wise
a @ b, torch.matmul(a, b), torch.bmm(b1, b2)        # matrix multiply
a.clamp(0, 1), a.abs(), a.sqrt(), a.exp(), a.log()
a.sum(), a.mean(), a.std(), a.max(), a.min()
a.argmax(dim=1), a.argmin(dim=0)
torch.cat([a, b], dim=0), torch.stack([a, b], dim=0)
torch.where(condition, x, y)
```

---

## Autograd API

```python
x = torch.tensor([1.0], requires_grad=True)
loss = (x ** 2).mean()
loss.backward()                           # compute gradients
x.grad                                    # access gradient (leaf only)
x.grad.zero_()                            # reset (done by optimizer.zero_grad())

with torch.no_grad():                     # disable tracking
    y = model(x)

with torch.inference_mode():              # faster alternative (1.9+)
    y = model(x)

tensor.detach()                           # break graph, no grad tracking
```

---

## nn.Module Template

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

model = MyModel()
model.to(device)
model.train(), model.eval()
model.parameters(), model.named_parameters()
model.state_dict(), model.load_state_dict(ckpt)
model.apply(init_fn)                      # apply to all sub-modules
```

### Common Layers

| Layer | Code |
|-------|------|
| Linear | `nn.Linear(in, out, bias=True)` |
| Conv2d | `nn.Conv2d(in_c, out_c, k, stride=1, padding=0)` |
| BatchNorm1d/2d | `nn.BatchNorm1d(num_features)` |
| Dropout | `nn.Dropout(p=0.5)` |
| LSTM | `nn.LSTM(input_size, hidden_size, num_layers)` |
| Embedding | `nn.Embedding(vocab_size, embed_dim)` |
| Flatten | `nn.Flatten()` |

### Activations

`nn.ReLU()`, `nn.GELU()`, `nn.Sigmoid()`, `nn.Tanh()`, `nn.Softmax(dim=1)`, `nn.LogSoftmax(dim=1)`

---

## Training Loop Skeleton

```python
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
scaler = torch.cuda.amp.GradScaler()              # for AMP

for epoch in range(num_epochs):
    # Train
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():            # optional AMP
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()              # loss.backward() if no AMP
        scaler.step(optimizer)                     # optimizer.step() if no AMP
        scaler.update()

    # Validate
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

    scheduler.step()
```

---

## Optimizer & Loss Summary

### Losses

| Loss | Task | Requirements |
|------|------|-------------|
| `CrossEntropyLoss` | Multi-class classification | Raw logits, class indices |
| `BCEWithLogitsLoss` | Binary / multi-label | Raw logits, 0/1 targets |
| `BCELoss` | Binary (manual) | Sigmoid outputs |
| `MSELoss` | Regression | Float targets |
| `L1Loss` | Regression (robust) | Float targets |
| `SmoothL1Loss` | Regression (Huber) | Float targets |
| `KLDivLoss` | Distribution matching | Log-probabilities |

### Optimizers

| Optimizer | Typical LR | Key Param |
|-----------|-----------|-----------|
| `SGD` | 0.01–0.1 | momentum=0.9 |
| `Adam` | 1e-3–3e-4 | betas=(0.9, 0.999) |
| `AdamW` | 1e-3–3e-4 | weight_decay=0.01 |
| `RMSprop` | 1e-3 | alpha=0.99 |

### Schedulers

| Scheduler | API |
|-----------|-----|
| `StepLR` | `.step()` — fixed drop every step_size |
| `ReduceLROnPlateau` | `.step(val_loss)` — monitors metric |
| `CosineAnnealingLR` | `.step()` — cosine decay |
| `CosineAnnealingWarmRestarts` | `.step()` — cosine with restarts |
| `LinearLR` | `.step()` — linear warmup |

---

## Save & Load Patterns

```python
# Save state_dict (recommended)
torch.save(model.state_dict(), "model.pt")

# Load state_dict
model.load_state_dict(torch.load("model.pt", weights_only=True))

# Save checkpoint (resume training)
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "loss": val_loss,
}, "checkpoint.pt")

# Load checkpoint
ckpt = torch.load("checkpoint.pt", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
scheduler.load_state_dict(ckpt["scheduler_state_dict"])
start_epoch = ckpt["epoch"] + 1

# TorchScript for production
scripted = torch.jit.script(model)
scripted.save("model_scripted.pt")
loaded = torch.jit.load("model_scripted.pt")
```
