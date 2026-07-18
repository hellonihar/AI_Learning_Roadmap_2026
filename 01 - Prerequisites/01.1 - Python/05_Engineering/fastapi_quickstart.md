# FastAPI Quickstart

## Basic App
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Inference API")

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    prediction: float
    confidence: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    result = model.predict([data.features])[0]
    return PredictResponse(
        prediction=float(result),
        confidence=0.95
    )
```

## Async Endpoint
```python
@app.post("/batch_predict")
async def batch_predict(data: list[PredictRequest]):
    features = [d.features for d in data]
    results = await async_model.predict(features)
    return [{"prediction": r} for r in results]
```

## Running
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
# Docs at http://localhost:8000/docs
```
