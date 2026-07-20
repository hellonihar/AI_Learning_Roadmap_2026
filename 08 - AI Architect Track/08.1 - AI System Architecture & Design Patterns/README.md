# AI System Architecture & Design Patterns

Master architectural styles, design patterns, and decision frameworks for building scalable, maintainable, and resilient AI systems.

## Learning Objectives
- Distinguish between monolithic, microservices, and event-driven architectures for AI workloads
- Design model-serving topologies that meet latency, throughput, and cost targets
- Apply proven design patterns to feature stores, model registries, and inference pipelines
- Document architectural decisions using ADRs

## Deep Topics

### 1. Distributed ML System Architectures
- End-to-end ML system topology: data ingestion → feature engineering → training → evaluation → deployment → monitoring → feedback loop
- Batch vs. real-time vs. streaming architectures; Lambda and Kappa architectures applied to ML
- Data mesh principles for ML: domain ownership, data as a product, self-serve infrastructure, federated governance
- Multi-region and multi-cluster deployments for global inference

### 2. Microservices vs. Monolith for AI Workloads
- When to decompose: model isolation, independent scaling, team boundaries
- Service mesh (Istio, Linkerd) for model traffic management: canary deployments, A/B testing, shadow traffic
- Sidecar patterns for model preprocessing, monitoring, and explainability
- Tradeoffs: network overhead vs. deployment flexibility; cold-start latency in serverless

### 3. Event-Driven Pipelines & Message Queues
- Asynchronous model invocation with Kafka, Pulsar, RabbitMQ
- Event sourcing for ML: tracking data changes, model retraining triggers, drift detection events
- Stream processing with Flink, Spark Streaming, or Kafka Streams for real-time feature computation
- Dead-letter queues, retry policies, and idempotency in inference pipelines

### 4. Model Serving Patterns
- **Online serving**: REST/gRPC endpoints, batching (dynamic/client-side), autoscaling (KPA, HPA)
- **Batch serving**: scheduled batch inference with Spark, MapReduce, or Beam
- **Streaming serving**: continuous inference on data streams with Flink or Kafka Streams
- **Edge serving**: on-device inference (TensorFlow Lite, Core ML, ONNX Runtime), offline-first sync
- **Hybrid patterns**: tiered serving with edge + cloud fallback

### 5. Feature Store Design & Patterns
- Online vs. offline feature stores: serving features for real-time inference vs. historical training
- Feature consistency: time-travel queries, point-in-time joins, feature freshness SLAs
- Technology choices: Feast, Tecton, SageMaker Feature Store, Azure Feature Store
- Feature cataloging, versioning, discovery, and lineage
- Streaming feature computation and materialization

### 6. Model Registry & Lifecycle Management
- Model versioning, lineage (training data, hyperparameters, code), and metadata (MLMD)
- Model promotion gates: staging validation, performance benchmarks, fairness checks
- Integration with CI/CD: automated registration, staging promotion, production rollback
- Artifact management: large model storage (S3, Blob, GCS), model signing, and integrity verification

### 7. Multi-Tenant Model Hosting
- Tenant isolation: dedicated vs. pooled model instances, resource quotas
- Cost attribution: chargeback/showback models per tenant
- Rate limiting, request prioritization, and SLA management across tenants

### 8. Architectural Decision Records (ADRs) for AI
- ADR template: context, decision, consequences, alternatives
- Sample ADRs: choosing batch vs. streaming, selecting feature store technology, designing for multi-cloud
- ADR review process and versioning alongside code

## Deliverables
- Design an online model-serving architecture for a multi-tenant SaaS platform
- Write ADRs for three major architectural decisions
- Compare batch vs. streaming architectures for a given use case with tradeoff analysis
