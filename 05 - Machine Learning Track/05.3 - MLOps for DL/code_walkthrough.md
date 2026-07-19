# Code Walkthrough: MLOps for Deep Learning

This walkthrough demonstrates a complete MLOps workflow for deep learning: distributed training with DDP and FSDP, checkpoint management, ONNX export, and inference optimisation.

---

## 1. Distributed Data Parallel (DDP) Training

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
        )

    def forward(self, x):
        return self.net(x)

def train_ddp(rank, world_size, epochs=10):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    model = SimpleModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        # Each rank processes a different micro-batch
        data = torch.randn(64, 1024).to(rank)
        target = torch.randn(64, 1024).to(rank)

        optimizer.zero_grad()
        output = ddp_model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size)
```

**Key points:**
- DDP wraps the model after moving it to the device.
- Gradients are automatically all-reduced across ranks inside `.backward()`.
- Only rank 0 prints/logs to avoid duplicate output.
- Launch with `torchrun --nproc_per_node=N train.py`.

---

## 2. FSDP Training with Checkpoint Sharding

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

def train_fsdp(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    model = SimpleModel().to(rank)
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=transformer_auto_wrap_policy,
        device_id=rank,
    )

    optimizer = optim.AdamW(fsdp_model.parameters(), lr=1e-4)

    for epoch in range(5):
        data = torch.randn(64, 1024).to(rank)
        target = torch.randn(64, 1024).to(rank)

        optimizer.zero_grad()
        loss = fsdp_model(data).sum()
        loss.backward()
        optimizer.step()

    # Save checkpoint — rank 0 consolidates full state dict
    save_fsdp_checkpoint(fsdp_model, optimizer, epoch=4, rank=rank)

    cleanup()
```

### Checkpoint Save (Sharded Approach)

```python
import torch.distributed.checkpoint as dcp

def save_fsdp_checkpoint(model, optimizer, epoch, rank):
    # Option A: Full state dict consolidation (rank 0 only)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state_dict = model.state_dict()
        if rank == 0:
            checkpoint = {
                "model": state_dict,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, f"fsdp_checkpoint_epoch{epoch}.pt")
            print(f"Rank {rank}: saved checkpoint")

    # Option B: Distributed checkpoint (each rank saves shard)
    # dcp.save(state_dict, checkpoint_id="path/to/checkpoint")
```

### Checkpoint Load

```python
def load_fsdp_checkpoint(model, optimizer, path, rank):
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        if rank == 0:
            checkpoint = torch.load(path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            epoch = checkpoint["epoch"]
        # Broadcast epoch to all ranks
        epoch_tensor = torch.tensor(epoch if rank == 0 else 0)
        dist.broadcast(epoch_tensor, src=0)

    # Alternative with DCP:
    # dcp.load(model.state_dict(), checkpoint_id="path/to/checkpoint")
```

---

## 3. ONNX Export and Inference Optimisation

### Export to ONNX

```python
import torch.onnx

def export_to_onnx(model, output_path="model.onnx"):
    model.eval()
    dummy_input = torch.randn(1, 1024)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"Model exported to {output_path}")
```

- `dynamic_axes` allows variable batch sizes at inference time.
- `opset_version=18` supports modern operators and INT8 quantization.

### ONNX Runtime Inference

```python
import numpy as np
import onnxruntime as ort

def run_onnx_inference(onnx_path="model.onnx", use_tensorrt=False):
    providers = ["CPUExecutionProvider"]
    if use_tensorrt:
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]

    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Benchmark
    import time
    data = np.random.randn(32, 1024).astype(np.float32)
    start = time.perf_counter()
    for _ in range(100):
        outputs = session.run([output_name], {input_name: data})
    elapsed = time.perf_counter() - start
    print(f"ONNX Runtime ({providers[0]}): {elapsed/100*1000:.2f} ms/batch")
```

### TensorRT Optimisation

```python
def build_tensorrt_engine(onnx_path="model.onnx", engine_path="model.trt"):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB
    config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(engine)
    print(f"TensorRT engine saved to {engine_path}")
```

### INT8 Quantization with ONNX Runtime

```python
def quantize_onnx_int8(onnx_path="model.onnx", output_path="model_int8.onnx"):
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        onnx_path,
        output_path,
        weight_type=QuantType.QInt8,
    )
    print(f"INT8 quantized model saved to {output_path}")

    # Size comparison
    import os
    fp32_size = os.path.getsize(onnx_path)
    int8_size = os.path.getsize(output_path)
    print(f"Size: FP32={fp32_size//1024**2}MiB → INT8={int8_size//1024**2}MiB "
          f"({int8_size/fp32_size*100:.0f}%)")
```

---

## 4. Full Pipeline: Training → Export → Deploy

```python
def full_mlops_pipeline():
    # Step 1: Train with DDP
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size, 10), nprocs=world_size)

    # Step 2: Load the best checkpoint on CPU
    model = SimpleModel()
    checkpoint = torch.load("best_model.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Step 3: Export to ONNX
    export_to_onnx(model, "model.onnx")

    # Step 4: Quantize to INT8
    quantize_onnx_int8("model.onnx", "model_int8.onnx")

    # Step 5: Build TensorRT engine
    build_tensorrt_engine("model_int8.onnx", "model.trt")

    # Step 6: Benchmark
    run_onnx_inference("model_int8.onnx", use_tensorrt=True)
```

---

## Summary

This walkthrough demonstrated:

| Step | Technique | Library |
|------|-----------|---------|
| 1 | Distributed training | PyTorch DDP |
| 2 | Sharded training | PyTorch FSDP |
| 3 | Checkpoint save/load | `torch.save` / DCP |
| 4 | Model export | ONNX |
| 5 | Quantization | ONNX Runtime quant |
| 6 | GPU optimisation | TensorRT |
| 7 | Inference serving | ONNX Runtime |

Each component can be swapped (e.g., DeepSpeed instead of FSDP, vLLM instead of ONNX for LLMs) but the pipeline structure remains the same.
