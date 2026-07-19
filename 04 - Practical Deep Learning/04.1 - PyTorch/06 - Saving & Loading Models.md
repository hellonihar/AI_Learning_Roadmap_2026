# Saving & Loading Models

Persisting and restoring model state is essential for training resumption, deployment, and sharing.

## state_dict: The Recommended Approach

Every `nn.Module` has a `state_dict` — a Python dict mapping parameter names to tensors.

```python
# Save
torch.save(model.state_dict(), "model.pth")

# Load
model = MyModel()
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.to(device)
```

**Why `state_dict`?** It saves only the learned parameters, not the model class definition. Your model code defines the architecture; the `state_dict` provides the weights. This is portable across versions and platforms.

## Full Model (Checkpoint)

For training resumption, save everything:

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_val_loss": best_val_loss,
    "model_arch": model.__class__.__name__,
}
torch.save(checkpoint, "checkpoint_epoch_10.pt")

# Load
ckpt = torch.load("checkpoint_epoch_10.pt", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
scheduler.load_state_dict(ckpt["scheduler_state_dict"])
start_epoch = ckpt["epoch"] + 1
```

Checkpoints are crucial for long training runs — save after every epoch (or N batches) and keep the best N.

## Save & Load Entire Model (Not Recommended)

```python
# Save — serializes the entire model object
torch.save(model, "model_full.pt")

# Load — requires the exact same class definition available in scope
model = torch.load("model_full.pt", weights_only=False)
```

**Avoid this.** It's brittle: changes to the model class, file paths, or PyTorch version break loading.

## TorchScript for Production

`torch.jit.script` or `torch.jit.trace` produces a self-contained, deployable representation:

```python
# Scripting — works with arbitrary control flow
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# Tracing — works with fixed input shapes, no control flow
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")

# Load anywhere (no Python class definition needed)
loaded = torch.jit.load("model_scripted.pt")
output = loaded(input_tensor)
```

TorchScript models run without Python dependencies — usable in C++ (libtorch), on mobile, and in production servers.

## File Format Overview

| Extension | Content | Use Case |
|-----------|---------|----------|
| `.pth` | `state_dict` dict | Standard weight saving |
| `.pt` | `state_dict` or full model | Same as `.pth`, interchangeable |
| `.ckpt` | Full checkpoint dict | Training resumption |
| `.pt` (scripted) | TorchScript binary | Production / C++ inference |
| `.onnx` | ONNX graph | Cross-framework inference |

**`.pt` vs `.pth`** — There is no technical difference. Both are just PyTorch-serialized files (pickle under the hood). The `.pth` convention was more common early on; `.pt` is now preferred to avoid confusion with Python path files.

## ONNX Export

```python
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}},
)
```

ONNX enables inference across frameworks (TensorRT, ONNX Runtime, OpenVINO). The `dynamic_axes` arg allows variable batch sizes.

## Best Practices

1. **Always save `state_dict`, not the full model.**
2. **Save checkpoints periodically** during long training — files are cheap; lost progress is expensive.
3. **Use `weights_only=True`** in `torch.load` when loading untrusted files — prevents arbitrary code execution.
4. **Keep the model definition in version control.** The `state_dict` is useless without matching architecture code.
5. **Save optimizer state** if you plan to resume training.
6. **Log metrics with the checkpoint** — epoch, validation loss, learning rate — so you know exactly what you're loading.

```python
# Saving shorthand: just state_dict
torch.save(model.state_dict(), f"model_epoch_{epoch:03d}.pt")

# Loading with strict=False allows partial loading (e.g., fine-tuning)
model.load_state_dict(torch.load("pretrained.pt", weights_only=True), strict=False)
```

> **Key takeaway:** `state_dict` is the standard. Add optimizer/scheduler state for resumable checkpoints. Use TorchScript for production deployment.
