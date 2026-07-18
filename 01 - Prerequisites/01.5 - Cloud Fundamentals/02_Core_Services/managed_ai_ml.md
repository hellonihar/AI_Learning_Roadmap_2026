# Managed AI/ML Services

| Capability | AWS | GCP | Azure |
|---|---|---|---|
| Full ML Platform | SageMaker | Vertex AI | Azure ML |
| Managed Notebooks | SageMaker Studio | Vertex AI Workbench | Azure ML Notebooks |
| Training | SageMaker Training | Vertex AI Training | Azure ML Compute |
| Hyperparameter Tuning | SageMaker HPO | Vertex AI Vizier | Azure ML Sweep |
| Model Registry | SageMaker Model Registry | Vertex AI Model Registry | Azure ML Model Registry |
| Endpoint Deployment | SageMaker Endpoints | Vertex AI Endpoints | Azure ML Endpoints |

## When to Use Managed ML vs DIY
- **Managed ML**: standard workflows (train → eval → deploy), less infra overhead
- **DIY (VMs + custom scripts)**: custom training loops, unsupported frameworks, full control

For most AI engineers, managed ML services handle 80% of use cases.
