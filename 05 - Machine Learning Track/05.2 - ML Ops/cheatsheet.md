# ML Ops Cheatsheet

## Data Versioning (DVC)

| Command                             | Purpose                          |
| ----------------------------------- | -------------------------------- |
| `dvc init`                          | Initialize DVC in repo           |
| `dvc add data/`                     | Track directory/file             |
| `dvc remote add -d myremote s3://b` | Configure remote storage         |
| `dvc push` / `dvc pull`             | Upload / download data           |
| `dvc repro`                         | Reproduce pipeline               |
| `dvc metrics diff`                  | Compare metrics across versions  |
| `dvc params diff`                   | Compare parameters               |
| `dvc checkout`                      | Restore data from cache          |

**Key files**: `.dvc` (pointer), `dvc.lock` (pipeline state), `dvc.yaml` (pipeline def)

## ML Pipelines

| Tool      | Config File          | DAG Definition                         | Best For                       |
| --------- | -------------------- | -------------------------------------- | ------------------------------ |
| Airflow   | `dags/*.py`          | `DAG()` context manager                | Batch scheduling, enterprise   |
| Kubeflow  | `pipeline.yaml`      | `@dsl.pipeline` decorator              | Kubernetes-native teams        |
| Prefect   | `flow.py`            | `@flow` / `@task` decorators           | Python-first, local dev        |
| Dagster   | `assets.py`          | `@asset` decorator                     | Data-product lineage           |

**Common DAG**: `ingest → validate → featurize → train → evaluate → register → deploy`

## Docker

| Concept          | Description                                        |
| ---------------- | -------------------------------------------------- |
| Image            | Read-only template (OS + deps + code)              |
| Container        | Runnable instance of an image                      |
| Dockerfile       | Instructions to build an image                     |
| docker-compose   | Multi-container orchestration                       |

**Dockerfile layers (order matters)**:
1. `FROM` — base image
2. `RUN apt-get` — system deps
3. `COPY requirements.txt` + `RUN pip install` — Python deps
4. `COPY src/ ./` — code
5. `CMD` — entrypoint

**Best practices**: multi-stage builds, non-root user, `.dockerignore`, pin tags

## Monitoring

| Drift Type    | What Changes         | Detection Method                    |
| ------------- | -------------------- | ----------------------------------- |
| Data drift    | P(X)                 | KS test, PSI, Chi-squared           |
| Concept drift | P(y\|X)              | Performance tracking, DDM, ADWIN   |
| Prediction drift | P(ŷ)              | PSI on prediction distribution      |
| Performance   | Metrics              | Accuracy, F1, RMSE over time        |

**Tools**: Evidently (reports), whylogs (profiles), Prometheus (metrics), Grafana (dashboards)

**Alert tiers**: Debug (PSI > 0.2) → Warning (data drift) → Critical (F1 drop)

## Experiment Tracking

| Tool     | Tracking Type       | Registry | Sweeps |
| -------- | ------------------- | -------- | ------ |
| MLflow   | Open source         | Yes      | Basic  |
| W&B      | SaaS / self-hosted  | Via artifacts | Excellent |
| Neptune  | SaaS                | Yes      | Good   |
| Comet    | SaaS                | Yes      | Good   |

**Log always**: params (lr, batch_size), metrics (acc, loss), artifacts (model, plots), env (pip freeze, git commit)

## Model Serving

| Pattern     | Latency     | Best For                        |
| ----------- | ----------- | ------------------------------- |
| Batch       | min–hours   | Offline scoring, nightly jobs   |
| Online REST | < 100 ms    | Real-time APIs                  |
| Online gRPC | < 10 ms     | High-performance, large payloads |
| Serverless  | 100 ms–10 s | Spiky workloads, prototypes     |

**Tools**: BentoML (easiest), FastAPI (flexible), TorchServe (PyTorch), TF Serving (TF), Triton (multi-framework, GPU)

## CI/CD for ML

**Pipeline stages**: `lint → test → data validation → train → evaluate → [gate] → promote → deploy`

**Data tests** (Great Expectations):
- `expect_column_values_to_not_be_null`
- `expect_column_min_to_be_between`
- `expect_column_distinct_values_to_be_in_set`

**Model tests**: invariance, performance threshold (acc > 0.85), fairness, robustness

**Conditional promotion**: `if metrics["f1"] > threshold then promote`

## Retraining

| Trigger        | When                           |
| -------------- | ------------------------------ |
| Time-based     | Every N days/weeks             |
| Performance    | Metric drops below threshold   |
| Drift-triggered | Data/concept drift detected   |

| Strategy       | Pros                     | Cons                    |
| -------------- | ------------------------ | ----------------------- |
| Full retrain   | Simple, safe             | Expensive, slow         |
| Incremental    | Fast, cheap              | Catastrophic forgetting |
| Windowed       | Adapts to recent trends  | Loses old patterns      |

**Validation**: Shadow deployment (parallel, no user impact) → A/B test (split traffic, measure business metrics)

## Quick Command Reference

```bash
# DVC
dvc add data/ && git add data.dvc && git commit -m "data v2" && dvc push

# Docker
docker build -t myapp:latest . && docker run -p 8080:8080 myapp

# MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
mlflow run . --experiment-name myexp

# Prefect
prefect deploy pipeline.py:ml_pipeline --name prod --cron "0 6 * * 1"

# Docker Compose
docker-compose up -d
```
