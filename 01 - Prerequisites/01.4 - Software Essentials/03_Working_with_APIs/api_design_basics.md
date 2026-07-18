# API Design Basics (FastAPI)

APIs are the primary interface through which AI models deliver value. A model sitting on a GPU server is useless without a well-designed API that lets applications consume its predictions. Every major AI product — ChatGPT, Claude, GitHub Copilot — is powered by APIs under the hood. An API wraps the complexity of model inference into a simple contract: send input, receive output.

For AI engineers, API design is as important as model performance. A 90% accurate model with a poorly designed API will fail in production; a 80% accurate model with a clean, reliable, well-documented API can be a successful product. The API defines how latency is handled, how errors are communicated, how authentication works, and how clients integrate. Getting these details right determines whether your model gets adopted or ignored.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="AI Service")

# Request / Response models
class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-4"
    temperature: float = 0.7

class ChatResponse(BaseModel):
    reply: str
    tokens_used: int

# Endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        response = await openai_client.chat.completions.create(
            model=req.model,
            messages=[{"role": "user", "content": req.message}],
            temperature=req.temperature,
        )
        return ChatResponse(
            reply=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
def health():
    return {"status": "ok", "model": "gpt-4"}
```

## API Versioning
```python
@app.get("/v1/predict")    # /v1/predict
@app.get("/v2/predict")    # /v2/predict (breaking changes)
```

## Environment-based Config
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    model_name: str = "gpt-4"
    rate_limit: int = 60

    class Config:
        env_file = ".env"
```
