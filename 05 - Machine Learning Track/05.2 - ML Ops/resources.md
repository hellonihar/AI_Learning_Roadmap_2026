# ML Ops Resources

## Tools

### Data Versioning
| Tool     | Description                                    | Link                                |
| -------- | ---------------------------------------------- | ----------------------------------- |
| DVC      | Open-source data versioning + pipelines        | https://dvc.org                     |
| Dolt     | Version-controlled SQL database                | https://dolthub.com                 |
| LakeFS   | Git-like branching on object storage           | https://lakefs.io                   |
| Quilt    | Data package manager for Python + S3           | https://quiltdata.com               |
| Hugging Face Datasets | Dataset library + versioned hub      | https://huggingface.co/datasets     |

### Pipeline Orchestration
| Tool     | Description                                    | Link                                |
| -------- | ---------------------------------------------- | ----------------------------------- |
| Airflow  | Industry standard batch orchestrator           | https://airflow.apache.org          |
| Prefect  | Modern Python-native DAG framework             | https://prefect.io                  |
| Dagster  | Asset-aware orchestration                      | https://dagster.io                  |
| Kubeflow | ML-specific pipelines on Kubernetes            | https://kubeflow.org                |
| Flyte    | Typed, multi-tenant ML orchestration           | https://flyte.org                   |
| Metaflow | Netflix's ML OSS framework                     | https://metaflow.org                |

### Containerization & Serving
| Tool        | Description                                    | Link                                |
| ----------- | ---------------------------------------------- | ----------------------------------- |
| Docker      | Container runtime                              | https://docker.com                  |
| Docker Compose | Multi-service orchestration                | https://docs.docker.com/compose     |
| BentoML     | Model serving framework                        | https://bentoml.com                 |
| TorchServe  | PyTorch model serving                          | https://pytorch.org/serve           |
| TF Serving  | TensorFlow model serving                       | https://tensorflow.org/tfx/guide/serving |
| Triton      | NVIDIA multi-framework inference server        | https://developer.nvidia.com/triton-inference-server |
| Ray Serve   | Scalable model serving on Ray                  | https://docs.ray.io/en/latest/serve |
| Seldon Core | ML deployment on Kubernetes                    | https://seldon.io                   |

### Experiment Tracking
| Tool     | Description                                    | Link                                |
| -------- | ---------------------------------------------- | ----------------------------------- |
| MLflow   | Open-source tracking + registry                | https://mlflow.org                  |
| W&B      | Deep learning experiment platform              | https://wandb.ai                    |
| Neptune  | Metadata ML platform for teams                 | https://neptune.ai                  |
| Comet    | Auto-logging experiment management             | https://comet.com                   |
| Aim      | Open-source visual experiment tracker          | https://aimstack.io                 |
| Sacred   | Python experiment framework                    | https://github.com/IDSIA/sacred     |

### Monitoring
| Tool        | Description                                    | Link                                |
| ----------- | ---------------------------------------------- | ----------------------------------- |
| Evidently   | Open-source drift & performance monitoring     | https://evidentlyai.com             |
| whylogs     | Open-source data logging                       | https://whylabs.ai/whylogs          |
| WhyLabs     | SaaS ML observability                          | https://whylabs.ai                  |
| Arize       | ML observability platform                      | https://arize.com                   |
| Prometheus  | Metrics collection and alerting                | https://prometheus.io               |
| Grafana     | Metrics visualization                          | https://grafana.com                 |
| Alibi Detect | Open-source drift detection library           | https://github.com/SeldonIO/alibi-detect |
| NannyML     | Post-deployment performance monitoring         | https://nannyml.com                 |

### CI/CD
| Tool          | Description                                    | Link                                |
| ------------- | ---------------------------------------------- | ----------------------------------- |
| GitHub Actions | CI/CD integrated with GitHub                 | https://github.com/features/actions |
| GitLab CI     | CI/CD integrated with GitLab                   | https://docs.gitlab.com/ee/ci      |
| Jenkins       | Extensible CI/CD server                        | https://jenkins.io                  |
| Great Expectations | Data validation and testing              | https://greatexpectations.io        |
| DVC           | (also used in CI for data pipeline testing)    | https://dvc.org                     |

### Feature Stores
| Tool       | Description                                    | Link                                |
| ---------- | ---------------------------------------------- | ----------------------------------- |
| Feast      | Open-source feature store                      | https://feast.dev                   |
| Tecton      | Enterprise feature platform                   | https://tecton.ai                   |
| Hopsworks  | Feature store + platform                       | https://hopsworks.ai                |

## Courses

| Course                                                           | Provider       | Link                                                          |
| ---------------------------------------------------------------- | -------------- | ------------------------------------------------------------- |
| MLOps Zoomcamp                                                   | DataTalksClub  | https://github.com/DataTalksClub/mlops-zoomcamp               |
| Machine Learning Engineering for Production (MLOps)              | DeepLearning.AI / Andrew Ng | https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops |
| Full Stack Deep Learning                                         | Full Stack Deep Learning | https://fullstackdeeplearning.com/course/              |
| MLOps (Coursera)                                                 | Duke University | https://www.coursera.org/specializations/mlops-machine-learning-operations |
| Data Engineering, Big Data, and ML on GCP                        | Google Cloud   | https://www.coursera.org/specializations/data-engineering-google-cloud |
| AWS Machine Learning Specialty                                   | AWS            | https://aws.amazon.com/certification/certified-machine-learning-specialty |
| Made With ML                                                    | Goku Mohandas  | https://madewithml.com                                       |

## Books & Papers

| Title                                                                  | Author(s)                   | Type    |
| ---------------------------------------------------------------------- | --------------------------- | ------- |
| Designing Data-Intensive Applications                                 | Martin Kleppmann            | Book    |
| Building Machine Learning Pipelines                                   | Hannes Hapke, Catherine Nelson | Book |
| Introducing MLOps                                                    | Mark Treveil et al.         | Book    |
| The ML Test Score: A Rubric for ML Production Systems                  | Eric Breck et al. (NeurIPS 2019) | Paper |
| Hidden Technical Debt in Machine Learning Systems                      | D. Sculley et al. (NeurIPS 2015) | Paper |
| A Chat with Andrew on MLOps: From Model-centric to Data-centric AI    | Andrew Ng (DeepLearning.AI) | Video   |
| Software Engineering for Machine Learning: A Case Study                | Arpteg et al. (ICSE 2018)   | Paper   |
| Continuous Delivery for Machine Learning                               | Sato et al. (2021)          | Paper   |

## Communities

| Community             | Platform | Link                                   |
| --------------------- | -------- | -------------------------------------- |
| MLOps.community       | Slack    | https://mlops.community                |
| DataTalksClub         | Slack    | https://datatalks.club                 |
| MLOps subreddit       | Reddit   | https://reddit.com/r/mlops             |
| ML Engineering        | Discord  | https://discord.gg/ml-engineering      |

## Reference Architectures

- **Google MLOps Level 0/1/2**: https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
- **Uber's Michelangelo**: https://eng.uber.com/michelangelo
- **Netflix's Metaflow**: https://netflixtechblog.com/open-sourcing-metaflow-a-human-centric-framework-for-data-science-fa72e04a5d9
- **Airbnb's Bighead**: https://medium.com/airbnb-engineering/bighead-airbnbs-end-to-end-machine-learning-platform-8dd7ac64cc8b
- **AWS MLOps Lens**: https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens
