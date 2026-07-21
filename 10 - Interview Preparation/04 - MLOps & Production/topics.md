# MLOps & Production

## Deployment Strategies
- Real-time inference (REST/gRPC), batch inference, streaming
- Deployment patterns: shadow, canary, blue-green, A/B testing
- Edge deployment — ONNX, TensorRT, CoreML, TFLite
- Model as a service vs. embedded model

## CI/CD for ML
- Data and model validation gates
- Automated retraining pipelines, scheduled drift retraining
- Feature engineering CI/CD
- Model promotion workflows (staging → prod)

## Monitoring & Observability
- Prediction monitoring: data drift, concept drift, feature drift
- Performance monitoring: latency, throughput, error rates
- Model degradation detection
- Logging, tracing, alerting (Prometheus, Grafana, MLflow)

## Model Governance
- Model versioning, lineage tracking (DVC, MLflow, W&B)
- A/B experiment tracking
- Approval workflows for model promotion
- Audit trails, compliance reporting

## Infrastructure
- Kubernetes for ML — KubeFlow, Ray
- GPU orchestration, node pools, spot instances
- Cost management — GPU utilization optimization
- Serverless inference (AWS Lambda, Cloud Run)

## Common Interview Questions
- How do you detect and handle data drift in production?
- Design a CI/CD pipeline for an ML system with automated retraining.
- What metrics would you monitor for a production LLM service?
- How would you roll back a bad model deployment with zero downtime?
- Compare serverless vs. dedicated GPU serving for cost and latency.
