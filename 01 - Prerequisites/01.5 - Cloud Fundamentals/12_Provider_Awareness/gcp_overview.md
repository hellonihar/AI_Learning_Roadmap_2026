# Google Cloud Platform (GCP) Overview

## Key Services for AI
| Category | Service |
|---|---|
| Compute | GCE (GPU: a3, a2, g2), GKE, Cloud Run, Cloud Functions |
| Storage | GCS (object), Persistent Disk, Filestore |
| ML Platform | Vertex AI (training, endpoints, notebooks, AutoML) |
| AI Services | Vertex AI Agent Builder, Document AI, Vision API, Natural Language |
| Databases | Cloud SQL, Firestore, Bigtable, Memorystore |
| Networking | VPC, Cloud LB, Cloud DNS, Cloud CDN |
| Security | IAM, Cloud KMS, Secret Manager, Cloud Armor |
| CI/CD | Cloud Build, Cloud Deploy, Artifact Registry |

## Strengths
- **Vertex AI**: tightly integrated with GCP ecosystem
- **BigQuery**: serverless data warehouse — great for feature engineering at scale
- **Kubernetes**: GKE is the best managed Kubernetes offering
- **TPUs**: custom Tensor Processing Units for large-scale training (faster/cheaper than GPUs for some models)
- **Data & AI integration**: BigQuery + Vertex AI + Dataflow for end-to-end ML pipelines

## Considerations
- Smaller GPU selection than AWS (but good coverage)
- Less enterprise adoption than AWS/Azure in some sectors
- Documentation can be fragmented across products
