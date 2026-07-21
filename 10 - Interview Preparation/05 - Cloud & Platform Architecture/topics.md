# Cloud & Platform Architecture

## Cloud Providers for AI
- **AWS**: SageMaker, Bedrock, EKS for GPU, Inferentia/Trainium
- **Azure**: AI Foundry, Machine Learning, OpenAI Service, ND-series VMs
- **GCP**: Vertex AI, GKE with GPUs/TPUs, Model Garden
- Multi-cloud vs. single-cloud strategy

## AI Platform Components
- Data lakehouse: Databricks, Snowflake, Delta Lake
- Feature platform: Feast, Tecton
- Model registry & metadata store: MLflow, W&B, Neptune
- Orchestration: Airflow, Kubeflow Pipelines, Prefect, Dagster

## Compute & Networking
- GPU instance types & SKUs (A100, H100, H200, B200)
- Spot vs. on-demand vs. reserved vs. PTU
- Inter-node networking: InfiniBand, Elastic Fabric Adapter (EFA)
- Storage tiering: object (S3/ADLS/GCS) → ephemeral (NVMe) → memory

## Security & Identity
- RBAC, IAM roles, service principals
- VPC isolation, private endpoints, network encryption
- Secrets management (Vault, AWS Secrets Manager)
- Data encryption at rest and in transit

## Cost Optimization
- Reserved capacity vs. pay-as-you-go
- Autoscaling GPU clusters, spot instance fallback
- Model distillation for cost reduction
- Caching layers to reduce API costs

## Common Interview Questions
- Compare AWS Bedrock, Azure AI Foundry, and GCP Vertex AI.
- How would you design a multi-cloud disaster recovery strategy for AI workloads?
- What GPU SKU would you choose for fine-tuning a 70B model vs. serving it?
- How do you secure an AI pipeline across data ingestion, training, and inference?
- Walk through a cost-optimized architecture for a high-volume inference service.
