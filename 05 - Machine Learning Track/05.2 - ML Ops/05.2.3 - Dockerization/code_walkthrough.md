# Code Walkthrough: Dockerize a Model Inference API

This walkthrough builds a FastAPI inference service for a trained scikit-learn model, packages it with Docker, and runs it locally.

## Project Structure

```
ml-inference/
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── src/
│   ├── app.py          # FastAPI application
│   └── predict.py      # prediction helper
└── models/
    └── model.pkl       # trained model (joblib dump)
```

## Step 1: Prediction Helper

```python
# src/predict.py
import joblib
import numpy as np
from pathlib import Path


class Predictor:
    def __init__(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = joblib.load(model_path)
        self._feature_names = getattr(self.model, "feature_names_in_", None)

    def predict(self, features: list[list[float]]) -> list:
        X = np.array(features, dtype=np.float64)
        preds = self.model.predict(X)
        probs = getattr(self.model, "predict_proba", None)
        probabilities = probs(X).tolist() if probs else None
        return {
            "predictions": preds.tolist(),
            "probabilities": probabilities,
        }

    def health(self) -> dict:
        return {"status": "ok", "model_type": type(self.model).__name__}
```

## Step 2: FastAPI Application

```python
# src/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from predict import Predictor

app = FastAPI(title="ML Inference API", version="1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
predictor = Predictor(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[list[float]]


class PredictResponse(BaseModel):
    predictions: list
    probabilities: list | None = None


@app.get("/health")
def health():
    return predictor.health()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.features:
        raise HTTPException(status_code=400, detail="No features provided")
    result = predictor.predict(request.features)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

## Step 3: Dependencies

```txt
# requirements.txt
fastapi==0.109.0
uvicorn==0.27.0
scikit-learn==1.4.0
joblib==1.3.2
pydantic==2.5.0
numpy==1.26.0
```

## Step 4: Dockerfile

```dockerfile
# Stage 1: Builder — install dependencies
FROM python:3.10-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime — minimal image
FROM python:3.10-slim AS runtime

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code and model
WORKDIR /app
COPY src/ ./src/
COPY models/ ./models/

# Switch to non-root user
USER appuser

# Port and entrypoint
EXPOSE 8080
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

## Step 5: .dockerignore

```dockerignore
__pycache__/
.git/
.vscode/
*.pkl
!models/model.pkl
*.dvc
.dvc/
data/
notebooks/
.env
```

## Step 6: Sample Model

Generate a toy model for testing:

```python
# scripts/train_sample.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs("models", exist_ok=True)
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)
joblib.dump(model, "models/model.pkl")
print("Model saved to models/model.pkl")
```

## Step 7: Build and Run

```bash
# Train sample model
python scripts/train_sample.py

# Build the Docker image
docker build -t ml-inference:latest .

# Run the container
docker run -d --name ml-api -p 8080:8080 ml-inference:latest

# Test health endpoint
curl http://localhost:8080/health

# Test prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]}'

# Check logs
docker logs ml-api

# Stop the container
docker stop ml-api
docker rm ml-api
```

## Step 8: Docker Compose (Optional)

Extend with Redis caching and PostgreSQL logging:

```yaml
# docker-compose.yml
services:
  inference:
    build: .
    ports:
      - "8080:8080"
    environment:
      - MODEL_PATH=/app/models/model.pkl
      - REDIS_URL=redis://redis:6379
    depends_on:
      redis:
        condition: service_started

  redis:
    image: redis:alpine
```

Run: `docker-compose up -d`

## Expected Output

A successful prediction returns:

```json
{
  "predictions": [0, 2],
  "probabilities": [
    [0.98, 0.02, 0.0],
    [0.0, 0.04, 0.96]
  ]
}
```

## Key Takeaways

- The **multi-stage build** keeps the final image small (~150 MB vs 1 GB for a non-slim image).
- The **non-root user** prevents container escapes via application vulnerabilities.
- **Environment variables** (`MODEL_PATH`) make the image configurable without rebuilding.
- **Layers are cached** — `requirements.txt` rarely changes, so `pip install` is cached on most rebuilds.
