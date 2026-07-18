# Cloud Quick Reference

## Service Models
```
IaaS  → VMs, storage, networking   (you manage OS, runtime)
PaaS  → managed runtime, DB, queue  (you manage code only)
SaaS  → ready-to-use app            (you just use it)
```

## Deployment Types
```
Public   → shared infra, multi-tenant
Private  → dedicated infra, single-tenant
Hybrid   → connect on-prem + cloud
Multi    → use 2+ cloud providers
```

## Compute Options
```
VM (EC2/GCE)     → full OS control, any workload
Container (ECS)  → lightweight, portable, microservices
Serverless (Lambda) → event-driven, no infra management
```

## Storage Types
```
Object (S3/GCS)    → unlimited, HTTP-accessible, for data lakes
Block (EBS/PD)     → attach to VM, low-latency, DB storage
File (EFS/Filestore) → shared NFS, multi-VM access
```

## Key AI Services by Provider
| Service | AWS | GCP | Azure |
|---|---|---|---|
| ML Platform | SageMaker | Vertex AI | Azure ML |
| GPU Instances | p4d/p5 | A3/A2 | NCas/NDas |
| Serverless Inference | SageMaker Serverless | Vertex AI Endpoints | Managed Online Endpoints |
| Vector DB | OpenSearch Serverless | Vector Search | Cosmos DB vCore |

## IAM Best Practices
- Least privilege: grant only what's needed
- Use roles, never share long-lived keys
- Prefer service accounts over user accounts for apps
- Rotate secrets regularly
