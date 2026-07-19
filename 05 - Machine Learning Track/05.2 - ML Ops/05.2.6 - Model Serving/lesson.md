# 05.2.6 Model Serving

## Model Serving Patterns

Once a model is trained and registered, it must be **served** — made available for predictions. The serving strategy depends on latency requirements, throughput, cost, and infrastructure.

### 1. Batch Inference

Batch inference processes a large set of inputs on a schedule and writes predictions to a store (database, data warehouse, or file).

**Characteristics**:
- High throughput, high latency (minutes to hours).
- No request-response requirement.
- Efficient resource utilization (batching maximizes GPU/CPU).

**Use cases**:
- Nightly customer churn scoring (run on all active users).
- Monthly credit limit recalculation.
- Offline recommendation generation.

**Implementation**: A scheduled job (Airflow, Prefect, cron) that loads the model, reads data from a feature store, predicts, and writes results.

```python
# batch_inference.py
model = joblib.load("models/model.pkl")
df = spark.read.table("features.customer_churn")
predictions = model.predict(df.toPandas())
df["churn_probability"] = predictions
df.write.mode("overwrite").saveAsTable("predictions.churn_scores")
```

### 2. Online / Real-time Inference

Online inference serves individual predictions with low latency (milliseconds to seconds) via a REST or gRPC API.

**Characteristics**:
- Low latency (p99 < 100 ms for many applications).
- Handles request spikes (autoscaling).
- Always-available endpoints.

**Use cases**:
- Credit card fraud detection (predict in real time per transaction).
- Search ranking.
- Chatbots and LLM inference.
- Recommendation personalization.

**Implementation**: A web server (FastAPI, Flask) loads the model on startup and exposes `/predict` or `/v1/models/{model_name}:predict` endpoints.

```python
@app.post("/predict")
def predict(request: PredictRequest):
    features = preprocess(request.features)
    pred = model.predict(features)
    return {"prediction": pred.tolist()}
```

### 3. Serverless Inference

Serverless platforms (AWS Lambda, GCP Cloud Functions, Azure Functions) abstract infrastructure entirely.

**Characteristics**:
- Auto-scales to zero (no cost when idle).
- Cold start latency (1–10 seconds for model loading).
- Timeout limits (typically 15 minutes per invocation).
- Memory limits (10 GB max on Lambda).

**Use cases**:
- Low-traffic or spiky workloads.
- Prototypes and MVPs.
- Event-driven inference (predict when a file lands in S3).

**Caveats**: Serverless is generally unsuitable for large models (> 1 GB) or latency-sensitive applications. Use **Lambda with container images** or **SageMaker Serverless Inference** for slightly larger models.

### Comparison

| Pattern       | Latency       | Throughput   | Cost                     | Operational Complexity |
| ------------- | ------------- | ------------ | ------------------------ | ---------------------- |
| Batch         | Minutes/hours | Very high    | Low (per compute)        | Low                    |
| Online (REST) | < 100 ms      | Medium-high  | Medium (always on)       | Medium                 |
| Online (gRPC) | < 10 ms       | High         | Medium (always on)       | High                   |
| Serverless    | 100 ms–10 s   | Low-medium   | Pay-per-invocation       | Very low               |

## Serving Tools

### BentoML

An open-source framework that packages models with serving logic, dependencies, and a REST API into a **Bento** (standardized deployable artifact).

```python
import bentoml
from bentoml.io import JSON

@bentoml.service(name="iris_classifier")
class IrisClassifier:
    def __init__(self):
        self.model = bentoml.sklearn.load_model("iris_classifier:latest")

    @bentoml.api(input=JSON(), output=JSON())
    def predict(self, features):
        return self.model.predict(features).tolist()
```

BentoML handles batching, tracing, and adaptive batching automatically. Deploy to Docker, Kubernetes, AWS Lambda, or SageMaker with a single command.

### TorchServe

A performant serving solution from PyTorch. Supports model versioning, A/B testing, and metrics.

```python
# model_store/config.yaml
minWorkers: 1
maxWorkers: 4
batchSize: 8
maxBatchDelay: 100
```

TorchServe uses a **model store** directory structure: `model_store/my_model/1/model.pt`.

### TensorFlow Serving (TF Serving)

Purpose-built for TensorFlow models. Uses gRPC for extreme performance.

```bash
tensorflow_model_server \
  --model_base_path=/models \
  --model_name=my_model \
  --rest_api_port=8501
```

TF Serving supports **model versioning** (subdirectories named by integer version) and **warm-up requests** to pre-initialize the computation graph.

### Triton Inference Server

NVIDIA's production-grade inference server supporting multiple frameworks (TensorFlow, PyTorch, ONNX, TensorRT) in a single server. Supports GPU batching, dynamic batching, model ensembles, and concurrent model execution.

**Why Triton?**
- **Concurrent model execution** — serve multiple models on one GPU.
- **Dynamic batching** — automatically aggregate requests for GPU efficiency.
- **Model pipelines** — chain pre-processing, inference, post-processing without Python code.
- **GPU vs CPU scheduling** — route requests to available hardware.

```bash
tritonserver --model-repository=/models --allow-gpu-metrics=true
```

## gRPC vs REST

| Feature         | REST                    | gRPC                             |
| --------------- | ----------------------- | -------------------------------- |
| Protocol        | HTTP/1.1                | HTTP/2                            |
| Serialization   | JSON                    | Protocol Buffers (binary)        |
| Performance     | Good                    | 5–10x faster for large payloads  |
| Streaming       | No (SSE for partial)    | Bidirectional streaming          |
| Browser support | Native                  | Requires gRPC-Web                |
| Debugging       | curl, browser           | grpcurl, specialized tools       |

For most teams, **start with REST** and move to gRPC if latency requirements demand it.

## Best Practices

- **Pre-load the model** — load model weights at server startup, not on every request.
- **Use a separate process for preprocessing** — or at least ensure preprocessing is vectorized.
- **Set request timeouts** — avoid requests hanging forever on a slow model.
- **Implement health and readiness probes** — `/health`, `/ready` endpoints for orchestrators.
- **Enable request logging** — log inputs, predictions, and latency (but be careful with PII).
- **Autoscale based on request queue depth** — not just CPU/memory.

## Summary

Choose batch inference for offline workloads, online REST/gRPC for real-time applications, and serverless for spiky or prototype workloads. BentoML offers the easiest path from notebook to production, while Triton and TF Serving deliver maximum performance for high-throughput scenarios. Align the serving pattern with your latency budget and operational capacity.
