# Code Walkthrough: FastAPI Model Serving Endpoint

This walkthrough creates a production-ready inference API with FastAPI, loads a pre-trained scikit-learn model, serves predictions, and packages everything with Docker.

## Project Structure

```
model-serving/
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── src/
│   ├── app.py
│   ├── model.py        # model loading and prediction
│   └── schemas.py      # pydantic request/response models
└── models/
    └── model.pkl       # trained model (from previous walkthrough)
```

## Step 1: Schemas

```python
# src/schemas.py
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: list[list[float]] = Field(
        ...,
        example=[[5.1, 3.5, 1.4, 0.2]],
        description="List of feature vectors",
    )


class PredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[list[float]] | None = None
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
```

## Step 2: Model Loader

```python
# src/model.py
import joblib
import numpy as np
import mlflow
import os
from pathlib import Path


class InferenceModel:
    def __init__(self, model_path: str | None = None):
        self.model = None
        self.version = "unknown"
        self._load(model_path)

    def _load(self, model_path: str | None):
        if model_path is None:
            model_path = os.getenv("MODEL_PATH", "models/model.pkl")

        # Try MLflow registry first, then local file
        if model_path.startswith("models:/"):
            self.model = mlflow.sklearn.load_model(model_path)
            # Extract version from URI
            self.version = model_path.split("/")[-1]
        elif Path(model_path).exists():
            self.model = joblib.load(model_path)
            self.version = model_path
        else:
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Set MODEL_PATH env var or provide a valid path."
            )

    def predict(self, features: list[list[float]]) -> dict:
        X = np.array(features, dtype=np.float64)
        preds = self.model.predict(X)

        proba = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X).tolist()

        return {
            "predictions": preds.tolist(),
            "probabilities": proba,
            "model_version": self.version,
        }

    def is_loaded(self) -> bool:
        return self.model is not None
```

## Step 3: FastAPI Application

```python
# src/app.py
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from prometheus_client import Histogram, Counter, generate_latest

from schemas import PredictRequest, PredictResponse, HealthResponse
from model import InferenceModel

# --- Prometheus metrics ---
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent processing prediction requests",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
PREDICTION_COUNT = Counter(
    "prediction_total",
    "Total number of predictions",
    ["model_version"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model: InferenceModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Loading model...")
    model = InferenceModel()
    logger.info(f"Model loaded (version: {model.version})")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="ML Model Serving API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    if model is None or not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=model.version,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, http_request: Request):
    if model is None or not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.features:
        raise HTTPException(status_code=400, detail="Empty features")

    # Validate feature dimensions
    n_features = len(request.features[0])
    # Infer expected feature count from model
    expected = getattr(model.model, "n_features_in_", None)
    if expected and n_features != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected} features, got {n_features}",
        )

    start = time.time()
    try:
        result = model.predict(request.features)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.time() - start
    PREDICTION_LATENCY.observe(latency)
    PREDICTION_COUNT.labels(model_version=model.version).inc(len(request.features))

    logger.info(
        f"Predicted {len(request.features)} samples "
        f"in {latency*1000:.1f}ms "
        f"(client: {http_request.client.host})"
    )

    return PredictResponse(
        predictions=result["predictions"],
        probabilities=result["probabilities"],
        model_version=result["model_version"],
    )


@app.get("/metrics")
def metrics():
    return generate_latest()
```

## Step 4: Dependencies

```txt
# requirements.txt
fastapi==0.109.0
uvicorn==0.27.0
scikit-learn==1.4.0
joblib==1.3.2
pydantic==2.5.0
numpy==1.26.0
mlflow==2.10.0
prometheus-client==0.20.0
```

## Step 5: Dockerfile

```dockerfile
FROM python:3.10-slim

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

USER appuser

EXPOSE 8080

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

## Step 6: .dockerignore

```dockerignore
__pycache__/
.git/
.vscode/
*.pkl
!models/model.pkl
.env
notebooks/
```

## Step 7: Build and Run

```bash
# 1. Ensure you have a model
python ../05.2.5\ -\ Experiment\ Tracking/train.py
cp models/model.pkl .  # or train a sample one

# 2. Build image
docker build -t model-serving:latest .

# 3. Run container
docker run -d --name ml-serve -p 8080:8080 model-serving:latest

# 4. Health check
curl http://localhost:8080/health

# 5. Predict
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]}'

# 6. Prometheus metrics
curl http://localhost:8080/metrics

# 7. Interactive docs
# Open http://localhost:8080/docs in browser
```

## Expected Prediction Response

```json
{
  "predictions": [0, 2],
  "probabilities": [
    [0.98, 0.02, 0.0],
    [0.0, 0.04, 0.96]
  ],
  "model_version": "models/model.pkl"
}
```

## Step 8: Stress Test (Optional)

```bash
# Install locust
pip install locust

# locustfile.py
from locust import HttpUser, task

class ModelUser(HttpUser):
    @task
    def predict(self):
        self.client.post("/predict", json={
            "features": [[5.1, 3.5, 1.4, 0.2]]
        })

# Run
locust --host http://localhost:8080
```

## Key Takeaways

- **Clean separation**: schemas → model loader → API → Docker.
- **Health and metrics endpoints** are essential for production orchestration.
- **Model loaded at startup**, not per request — avoids cold-start latency.
- **Prometheus metrics** enable integration with Grafana dashboards.
- **Input validation** catches malformed requests before they reach the model.
- **Docker packaging** makes deployment identical across environments.
