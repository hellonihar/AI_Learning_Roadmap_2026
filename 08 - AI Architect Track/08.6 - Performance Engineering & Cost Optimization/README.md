# Performance Engineering & Cost Optimization

Optimize AI workloads for latency, throughput, memory efficiency, and cost across training and inference.

## Learning Objectives
- Apply model compression techniques (quantization, pruning, distillation) to reduce cost
- Profile and optimize GPU utilization for training and inference
- Implement FinOps practices for AI workload cost governance

## Deep Topics

### 1. Model Compression
- **Quantization**: PTQ (Post-Training Quantization) vs. QAT (Quantization-Aware Training), INT4/INT8/FP8/FP16
- **Pruning**: unstructured (magnitude pruning, SparseGPT) vs. structured (channel, layer) pruning
- **Knowledge distillation**: teacher-student training, logit matching, feature distillation
- **Combined approaches**: quantization + pruning + distillation for maximum compression

### 2. Inference Acceleration
- **Hardware-specific optimizations**: TensorRT (NVIDIA), OpenVINO (Intel), Core ML (Apple), ONNX Runtime
- **vLLM**: PagedAttention, continuous batching, prefix caching for LLM inference
- **TensorRT-LLM**: in-flight batching, multi-GPU multi-node inference
- **Sparsity**: 2:4 structured sparsity on Ampere+ GPUs, activation sparsity
- **Operator fusion**: fusing adjacent operations to reduce memory bandwidth
- **FlashAttention**: IO-aware exact attention for faster transformer inference

### 3. GPU/TPU Utilization & Profiling
- **Profiling tools**: NVIDIA Nsight Systems, Nsight Compute, PyTorch Profiler, TensorBoard
- **Key metrics**: GPU utilization, memory bandwidth utilization, compute vs. memory bound
- **Data loading bottlenecks**: parallel data loading, prefetching, caching, shared memory
- **Distributed training communication**: NCCL all-reduce, gradient compression, async SGD
- **TPU optimization**: XLA compilation, sharding, data parallelism, model parallelism

### 4. Batch vs. Streaming Inference Trade-offs
- **Dynamic batching**: window-based, pressure-based, adaptive batch sizing
- **Latency-throughput Pareto front**: finding optimal batch size for SLA constraints
- **Streaming inference**: one-at-a-time processing with stateful models (RNNs, online learning)
- **Preprocessing pipelining**: overlapping data preprocessing with computation

### 5. FinOps for AI
- **Cost dimensions**: compute (GPU/CPU), storage (data, models, artifacts), network (egress, inter-region)
- **Cost allocation**: tagging resources by project/team/model, chargeback/showback dashboards
- **Reserved/spot instances**: compute cost optimization with spot interruption handling
- **Budget alerts**: proactive notification when spend exceeds thresholds
- **Tools**: AWS Cost Explorer, Azure Cost Management, GCP Billing, Kubecost, Vantage

### 6. Capacity Planning & Autoscaling
- **Demand forecasting**: historical traffic patterns, seasonality, launch events
- **Autoscaling policies**: CPU/memory-based, request-based, custom metrics (queue depth, GPU utilization)
- **Node pool management**: heterogeneous pools (A100, H100, L4, T4) with node auto-provisioning
- **Over-provisioning strategies**: buffer for traffic spikes vs. cost waste tradeoff

### 7. Cost-per-Inference Analysis
- **Total cost of inference**: compute, memory, storage, network, serving infrastructure
- **Cost per token / per request / per user**: granular cost tracking
- **Model selection impact**: smaller model + more retrieval vs. larger model
- **Caching impact**: semantic caching hit rate on cost reduction

### 8. Energy Efficiency & Sustainability
- **Carbon footprint measurement**: power consumption per training run, per inference
- **Green computing**: training in regions with lower carbon intensity, using renewable energy
- **Hardware selection**: energy per FLOP comparisons across GPU generations
- **Model efficiency vs. accuracy tradeoff**: compute-optimal scaling (Chinchilla law)

## Deliverables
- Profile and optimize a model's inference latency, targeting p50 < 100ms and p99 < 500ms
- Build a FinOps dashboard tracking cost per model per environment with budget alerts
- Compare cost-per-1M-tokens across three model sizes and recommend the optimal tradeoff
