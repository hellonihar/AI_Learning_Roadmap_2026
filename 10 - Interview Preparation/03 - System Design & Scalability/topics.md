# System Design & Scalability

## End-to-End ML Pipeline Design
- Data ingestion, validation, feature engineering
- Training pipeline, hyperparameter tuning
- Model registry, versioning, governance
- Inference pipeline, serving, monitoring

## Distributed Training
- Data parallelism, model parallelism, pipeline parallelism
- FSDP (Fully Sharded Data Parallel), DeepSpeed ZeRO stages
- Gradient accumulation, checkpointing
- Multi-node training, network topology (NVLink, InfiniBand)

## Inference at Scale
- Batching strategies: dynamic batching, continuous batching
- Model parallelism for inference (tensor parallelism)
- Serving architectures: vLLM, Triton Inference Server
- Caching: KV cache, semantic cache, response cache
- Load balancing, auto-scaling, GPU utilization optimization

## Storage & Data
- Feature stores (Feast, Tecton)
- Vector databases — indexing (IVF, HNSW, DiskANN)
- Data lakehouse architecture (Delta Lake, Iceberg)
- Streaming vs. batch processing

## Trade-off Analysis Framework
- Latency vs. throughput vs. cost
- Accuracy vs. speed (model distillation, quantization)
- Consistency vs. availability (CAP theorem applied to ML)
- Build vs. buy — when to use SaaS APIs vs. self-host

## Common Interview Questions
- Design a real-time fraud detection system handling 10K QPS.
- How would you serve a 70B parameter model with <100ms latency?
- Design a feature store for a large-scale recommendation system.
- What breaks when your training data grows from 1TB to 10PB?
- How would you architect a multi-region ML deployment?
