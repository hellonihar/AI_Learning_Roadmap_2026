# Cloud AI Platform Architecture (Multi-cloud)

Design AI systems that can run across AWS, Azure, and Google Cloud while remaining cloud-agnostic and avoiding vendor lock-in.

## Learning Objectives
- Compare AI service offerings across major cloud providers and identify abstraction patterns
- Design infrastructure as code that can target multiple clouds
- Architect cost-optimized, resilient, multi-region AI deployments

## Deep Topics

### 1. Cloud AI Services Overview
- **AWS**: SageMaker (Studio, Pipelines, Model Registry, Feature Store), Bedrock, Rekognition, Comprehend, Lex
- **Azure**: Azure Machine Learning (Workspaces, Pipelines, Model Registry), Azure AI Services (Vision, Language, Speech), OpenAI Service, Cognitive Search
- **GCP**: Vertex AI (Pipelines, Model Registry, Feature Store, Vizier), AutoML, Natural Language, Vision, Speech-to-Text
- Service comparison matrix: capabilities, pricing models, integration depth

### 2. Abstraction Layers & Infrastructure as Code
- **Terraform / OpenTofu**: multi-cloud resource provisioning, modules for AI services
- **Pulumi**: infrastructure as code with general-purpose languages (Python, TypeScript, Go)
- **Crossplane**: control-plane-based infrastructure management for Kubernetes-native workflows
- **Abstraction strategies**: wrapper libraries, containerized tooling, Kubernetes as the common layer
- **Kubernetes** as the portability layer: KubeFlow, Ray, Kserve, Seldon Core across clouds

### 3. Hybrid & Edge Deployments
- On-premises / edge inference for low-latency or offline scenarios
- AWS Outposts, Azure Stack Edge, GCP Distributed Cloud Edge
- Hybrid ML pipelines: training in cloud, inference on edge; periodic sync of model updates
- Air-gapped environments: model registry replication, local training, batch upload

### 4. Cost-Optimized Compute Selection
- Spot/preemptible instances for training, reserved capacity for production inference
- Multi-cloud cost arbitrage: training in the cheapest region/cloud, inference closest to users
- GPU/TPU instance types across clouds: A100, H100, TPU v4/v5p, Inferentia, Habana
- Autoscaling policies: target utilization, schedule-based, prediction-based

### 5. Data Lake & Lakehouse Patterns Across Clouds
- Open table formats: Delta Lake, Apache Iceberg, Apache Hudi for multi-cloud data sharing
- Distributed data catalog: AWS Glue Catalog, Azure Purview, GCP Data Catalog, Apache Atlas
- Cross-cloud data replication: distcp, AWS DataSync, Azure Data Box, GCP Transfer Appliance
- Query engines: Presto/Trino, Athena, Redshift Spectrum, Synapse, BigQuery

### 6. Vendor Lock-in Mitigation Strategies
- Open standard adoption: ONNX, MLflow, Kubeflow, OCI containers
- Multi-cloud model registry: copy models across clouds for DR and latency optimization
- Data portability: store data in open formats on object storage accessible from any cloud
- Kubernetes as the control plane abstraction

### 7. Disaster Recovery & Multi-Region Failover
- Active-passive vs. active-active multi-region inference
- Global load balancers: AWS Route 53, Azure Traffic Manager, GCP Cloud Load Balancing
- Data replication strategies for model artifacts and feature stores
- DR testing: chaos engineering for multi-region AI deployments

### 8. Governance & Compliance Across Cloud Boundaries
- Consistent IAM roles across clouds using attribute-based access control (ABAC)
- Audit trail aggregation: AWS CloudTrail, Azure Monitor, GCP Cloud Audit Logs → centralized SIEM
- Data residency requirements: keep training/inference data in specific regions
- Compliance frameworks: SOC 2, HIPAA, PCI DSS across multi-cloud deployments

## Deliverables
- Design a multi-cloud AI architecture that can fail over between two cloud providers
- Write Terraform modules for deploying an ML pipeline on two different clouds
- Perform a cost comparison of training a large model on AWS vs. Azure vs. GCP
