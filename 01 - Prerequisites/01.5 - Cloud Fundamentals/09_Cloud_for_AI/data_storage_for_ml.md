# Data Storage for ML Pipelines

## Data Lake (Raw Data)
- Object storage (S3 / GCS) — cheap, durable, scalable
- Organize by: `project/dataset/date/file.parquet`
- Store raw, processed, and curated datasets separately

## Feature Store
- Centralized repository of features for training and serving
- Ensures training and inference use the same features
- **Managed**: SageMaker Feature Store, Feast, Tecton

## Artifact Registry
- Store model artifacts (`.pt`, `.pkl`, `.onnx`)
- Version with model registry (SageMaker Model Registry, MLflow)
- Tag with metadata: framework, accuracy, dataset used

## Experiment Tracking
- Log parameters, metrics, and artifacts for each run
- **Tools**: MLflow, Weights & Biases, TensorBoard
- Store logs/artifacts in S3 with experiment ID as path prefix

## File Organization Pattern
```
s3://ml-platform/
  raw-data/{project}/{date}/
  processed-data/{project}/{version}/
  features/{feature-group}/{version}/
  experiments/{experiment-id}/
    params.json
    metrics.json
    checkpoints/
  models/{model-name}/{version}/
    model.pt
    config.json
  predictions/{batch-job-id}/
```
