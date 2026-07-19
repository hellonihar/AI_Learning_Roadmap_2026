# Resources: MLOps for Deep Learning

---

## Papers

### Distributed Training

| Paper | Year | Key contribution |
|-------|------|-----------------|
| **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** (Rajbhandari et al.) | 2020 | ZeRO-1/2/3 sharding; foundation of FSDP and DeepSpeed |
| **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** (Shoeybi et al.) | 2020 | Tensor + pipeline parallelism for large transformer training |
| **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM** (Narayanan et al.) | 2021 | 3D parallelism (data + tensor + pipeline) |
| **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism** (Huang et al.) | 2019 | Pipeline parallelism with micro-batching |
| **Memory-Efficient Pipeline-Parallel DNN Training** (Narayanan et al.) | 2019 | PipeDream 1F1B scheduling |
| **PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel** (Zhao et al.) | 2023 | FSDP design and scaling results |
| **ZeRO-Offload: Democratizing Billion-Scale Model Training** (Ren et al.) | 2021 | Offloading optimizer states to CPU |
| **ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning** (Rajbhandari et al.) | 2021 | Offloading all states to CPU/NVMe |

### Quantization & Compression

| Paper | Year | Key contribution |
|-------|------|-----------------|
| **GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers** (Frantar et al.) | 2023 | Layer-wise INT4 quantization for LLMs |
| **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (Lin et al.) | 2024 | Activation-aware scaling for INT4 |
| **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** (Dettmers et al.) | 2022 | Mixed INT8/FP16 decomposition |
| **QLoRA: Efficient Finetuning of Quantized Language Models** (Dettmers et al.) | 2023 | NF4 quantization + LoRA |
| **Learning Intrinsic Low-Dimensionality of Deep Networks** (Distillation) | 2019 | Knowledge distillation theory |
| **DistilBERT, a Distilled Version of BERT** (Sanh et al.) | 2019 | Practical distillation for transformers |
| **Sparse GPU Kernels for Deep Learning** (Gale et al.) | 2020 | 2:4 structured sparsity |

### Inference

| Paper | Year | Key contribution |
|-------|------|-----------------|
| **Efficient Memory Management for Large Language Model Serving with PagedAttention** (Kwon et al.) | 2023 | PagedAttention; vLLM |
| **Orca: A Distributed Serving System for Transformer-Based Generative Models** (Yu et al.) | 2022 | Continuous / iteration-level batching |
| **FlexGen: High-Throughput Generative Inference of Large Language Models** (Sheng et al.) | 2023 | Offloading for throughput |
| **TensorRT: NVIDIA Inference Optimizer** (NVIDIA) | 2018+ | Layer fusion, kernel auto-tuning, INT8/FP16 |

### Checkpointing & Storage

| Paper | Year | Key contribution |
|-------|------|-----------------|
| **Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization** (Jain et al.) | 2020 | Recomputation vs checkpoint trade-off |
| **Training Giant Models with the Distributed Checkpointing of DeepSpeed** (DeepSpeed team) | 2022 | Sharded checkpointing at scale |

---

## Tools & Frameworks

### Distributed Training

| Tool | Description | Link |
|------|-------------|------|
| **PyTorch DDP** | Built-in data parallelism | pytorch.org/docs/stable/ddp.html |
| **PyTorch FSDP** | Fully sharded data parallel | pytorch.org/docs/stable/fsdp.html |
| **DeepSpeed** | ZeRO stages 1–3, offloading, MoE | github.com/microsoft/DeepSpeed |
| **Megatron-LM** | Tensor + pipeline parallelism | github.com/NVIDIA/Megatron-LM |
| **Megatron-DeepSpeed** | Combined 3D parallelism | github.com/microsoft/Megatron-DeepSpeed |
| **FairScale** | (Legacy) FSDP precursor | github.com/facebookresearch/fairscale |
| **Horovod** | Distributed training framework | github.com/horovod/horovod |
| **TorchElastic** | Fault-tolerant distributed launcher | pytorch.org/docs/stable/elastic/ |
| **NVIDIA NeMo** | Large model training framework | github.com/NVIDIA/NeMo |

### Inference

| Tool | Description | Link |
|------|-------------|------|
| **ONNX Runtime** | Cross-platform inference engine | onnxruntime.ai |
| **TensorRT** | NVIDIA GPU optimisation | developer.nvidia.com/tensorrt |
| **TensorRT-LLM** | LLM-optimised TensorRT | github.com/NVIDIA/TensorRT-LLM |
| **vLLM** | High-throughput LLM serving | github.com/vllm-project/vllm |
| **llama.cpp** | CPU-first LLM inference | github.com/ggerganov/llama.cpp |
| **CTranslate2** | Transformer inference engine | github.com/OpenNMT/CTranslate2 |
| **Triton Inference Server** | Production-grade serving | developer.nvidia.com/triton-inference-server |
| **HuggingFace TGI** | Text Generation Inference | github.com/huggingface/text-generation-inference |
| **OpenVINO** | Intel CPU/iGPU optimisation | docs.openvino.ai |

### Quantization & Compression

| Tool | Description | Link |
|------|-------------|------|
| **bitsandbytes** | INT8/NF4 quantization | github.com/TimDettmers/bitsandbytes |
| **AutoGPTQ** | GPTQ quantization | github.com/PanQiWei/AutoGPTQ |
| **AutoAWQ** | AWQ quantization | github.com/casper-hansen/AutoAWQ |
| **PyTorch AO** | `torch.ao.quantization` native API | pytorch.org/docs/stable/quantization.html |
| **NNCF** | Intel compression toolkit | github.com/openvinotoolkit/nncf |
| **TensorFlow Lite** | Mobile/edge optimisation | tensorflow.org/lite |
| **ExecuTorch** | PyTorch edge runtime | pytorch.org/executorch |

### Storage & Versioning

| Tool | Description | Link |
|------|-------------|------|
| **HuggingFace Hub** | Model registry, versioning | huggingface.co/docs/hub |
| **DVC** | Data version control | dvc.org |
| **Git LFS** | Large file storage for Git | git-lfs.com |
| **SafeTensors** | Safe, fast model serialisation | github.com/huggingface/safetensors |

---

## Official Documentation

| Topic | Link |
|-------|------|
| PyTorch Distributed Overview | pytorch.org/tutorials/intermediate/ddp_tutorial.html |
| PyTorch FSDP Tutorial | pytorch.org/tutorials/intermediate/FSDP_tutorial.html |
| DeepSpeed Getting Started | microsoft.github.io/DeepSpeed/ |
| ONNX Runtime Docs | onnxruntime.ai/docs/ |
| TensorRT Quick Start | developer.nvidia.com/quick-start-tensorrt |
| vLLM Documentation | vllm.readthedocs.io |
| NCCL Documentation | docs.nvidia.com/deeplearning/nccl/ |

---

## Courses & Talks

| Resource | Format | Link |
|----------|--------|------|
| **Stanford CS329S: ML Systems Design** | Course | cs329s.stanford.edu |
| **UC Berkeley CS267: Large-Scale ML** | Course | sites.google.com/berkeley.edu/cs267 |
| **DeepSpeed Tutorial (Microsoft)** | Video | youtube.com/watch?v=KStvD9v_XYk |
| **EfficientML.ai (MIT)** | Course | efficientml.ai |
| **NVIDIA GTC Talks on TensorRT** | Talks | nvidia.com/gtc |
| **PyTorch Conference Talks** | Talks | pytorch.org/community |
| **Scaling LLMs (Kilian Kluge)** | Blog | kluge.ai/scaling-llms |

---

## Tools for Further Practice

```bash
# Install core tools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install deepspeed onnx onnxruntime-gpu tensorrt
pip install vllm transformers accelerate bitsandbytes
pip install safetensors huggingface_hub dvc
pip install nvidia-ml-py3  # GPU monitoring

# Verify NCCL
python -c "import torch.distributed as dist; dist.init_process_group('nccl', rank=0, world_size=1); print('NCCL OK')"
```

---

## Contribution Guide

If a resource is broken or outdated, please open an issue at the repository for this learning roadmap. We aim to keep this list current with the rapidly evolving MLOps ecosystem.
