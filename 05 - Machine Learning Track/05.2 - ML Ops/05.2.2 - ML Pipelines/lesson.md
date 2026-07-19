# 05.2.2 ML Pipelines

## Pipeline Concepts

An ML pipeline is a directed acyclic graph (DAG) of processing steps that transforms raw data into a trained, evaluated, and registered model. Each step is a discrete, repeatable unit of work with explicit inputs and outputs.

### The Canonical DAG

```
Raw Data ──► Data Validation ──► Featurization ──► Training ──► Evaluation ──► Model Registration
                          │                                                        │
                          └──► Test Data ──────────────────────────────────────────┘
```

A well-designed pipeline makes the following guarantees:

- **Determinism** — same inputs produce the same outputs (fixed seeds, deterministic algorithms).
- **Isolation** — steps don't share state except through declared artifacts.
- **Caching** — steps whose inputs haven't changed are skipped.
- **Reusability** — any step can be swapped (e.g., a different featurization module) without touching the rest of the pipeline.
- **Observability** — every step logs parameters, metrics, and artifacts.

### Pipeline Steps in Detail

| Step            | Input                      | Output                  | Purpose                                      |
| --------------- | -------------------------- | ----------------------- | -------------------------------------------- |
| Data Ingestion  | Raw files, DB exports      | Parquet files           | Pull data from sources into a standard format |
| Validation      | Parquet files, schema      | Validated dataset       | Reject schema violations, null checks         |
| Featurization   | Validated dataset          | Feature matrix (X, y)   | Encode, scale, split, create derived features |
| Training        | Feature matrix, hyperparams| Model artifact          | Fit model, log params, save weights          |
| Evaluation      | Model, test set            | Metrics, plots          | Compute accuracy, precision, recall, etc.    |
| Registration    | Model, metrics             | Model registry entry    | Version and tag the model                     |
| Deployment      | Registered model           | Deployed endpoint       | Serve model in production                     |

## Orchestration Tools

Different tools suit different scales and preferences:

### Apache Airflow

The industry standard for batch orchestration. Pipelines are defined as **DAGs in Python**:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG("ml_pipeline", schedule="@weekly") as dag:
    ingest = PythonOperator(task_id="ingest", python_callable=ingest_data)
    validate = PythonOperator(task_id="validate", python_callable=validate_data)
    train = PythonOperator(task_id="train", python_callable=train_model)
    ingest >> validate >> train
```

Airflow excels at scheduling, retries, alerting, and dependency management. It is less suited for tight ML integrations (GPU scheduling, experiment tracking) without custom plugins.

### Kubeflow Pipelines

Purpose-built for ML on Kubernetes. Each step runs in a container, and the DAG is defined via a Python SDK. Kubeflow natively handles GPU allocation, experiment tracking, and model serving.

```python
from kfp import dsl

@dsl.component(base_image="python:3.10")
def train_op(data: str) -> str:
    # training code
    return model_path

@dsl.pipeline
def ml_pipeline(data: str):
    train_op(data)
```

Kubeflow is the best choice for organizations already on Kubernetes. Its tight coupling with K8s is both a strength (resource management) and a weakness (operational complexity).

### Prefect

A modern orchestrator that bridges the gap between Airflow and Kubeflow. Prefect offers Python-native DAGs, automatic retries, caching, and a dashboard. It runs on any infrastructure (local, K8s, serverless).

```python
from prefect import flow, task

@task
def featurize(data): ...

@task
def train(features): ...

@flow
def ml_pipeline(data):
    features = featurize(data)
    model = train(features)
    return model
```

Prefect's key advantages: **easy local development**, **automatic caching** (`cache_key_fn`), and a **free tier** for small teams.

### Dagster

Focuses on **asset-awareness** — it tracks data assets and their lineage, making it easy to understand what produced what.

```python
from dagster import asset, MaterializeResult

@asset
def cleaned_data(raw_data): ...

@asset
def model(cleaned_data): ...
```

Dagster is excellent when your pipeline complexity comes from many interconnected data products rather than compute-heavy training.

## Pipeline Determinism and Reproducibility

A pipeline is reproducible if, given the same code, data, and parameters, it produces the same model. Practical steps:

1. **Pin dependencies** — Use a lock file (`requirements.txt` hash, `poetry.lock`, `conda-lock.yml`) to guarantee the same library versions.
2. **Seed all randomness** — `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`. Also seed data splits.
3. **Use deterministic algorithms** — e.g., `torch.use_deterministic_algorithms(True)` (may be slower).
4. **Cache intermediate artifacts** — If step 1–3 haven't changed, skip to step 4.
5. **Log the full environment** — Save `pip freeze` or conda env export as an artifact.

### Non-Determinism Sources

- GPU operations (CUDA non-determinism)
- Hash randomization (Python `PYTHONHASHSEED`)
- Floating-point accumulation order (parallel reductions)
- Data shuffling without a fixed seed
- External API calls (time-varying features)

## Best Practices

- **Keep steps idempotent** — running a step twice with the same inputs should produce the same outputs.
- **Separate configuration from code** — hyperparameters, datasource paths, and flags should be external (YAML/JSON/env).
- **Version every output** — data, features, models, and metrics should all carry version tags.
- **Alert on failure** — configure notifications (Slack, PagerDuty) for pipeline failures.
- **Start simple** — a plain Makefile or Python script with `dvc repro` often beats a heavy orchestrator for small teams.

## Summary

ML pipelines are the backbone of reproducible, production-grade ML. Selecting the right orchestrator depends on your infrastructure (Airflow for Hadoop shops, Kubeflow for K8s shops, Prefect for Python-first teams). Regardless of tooling, the principles are universal: make each step deterministic, cache aggressively, and maintain a complete lineage from raw data to deployed model.
