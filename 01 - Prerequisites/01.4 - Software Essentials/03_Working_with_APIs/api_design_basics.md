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

---

## Alternatives to FastAPI

| Framework | Language | Style | Key Strength | Best For |
|---|---|---|---|---|
| **FastAPI** | Python | Async, Pydantic, OpenAPI | Auto-docs, type validation, async | Most AI APIs |
| **Flask** | Python | Sync, extensions | Simplicity, largest ecosystem | Small services, prototypes |
| **Django REST Framework** | Python | Full-stack, batteries-included | ORM, admin, authentication | Large web apps with AI features |
| **Litestar** | Python | Async, Pydantic, DI | Performance, DTOs | High-throughput APIs |
| **Quart** | Python | Async (Flask-compatible) | Flask syntax + async | Migrating Flask to async |
| **Express.js** | JavaScript | Async, middleware | Node.js ecosystem, speed | Real-time streaming (SSE/WebSocket) |
| **Bun Elysia** | TypeScript | Async, Elysia | ~15x faster than Express | Latency-critical APIs |
| **Go net/http** | Go | Sync, standard library | ~30x faster than Python, low memory | High-throughput, low-cost serving |

### Key Trade-offs

| Requirement | Recommended |
|---|---|
| Rapid prototyping with Python | FastAPI or Flask |
| Auto-generated OpenAPI docs | FastAPI (built-in), Flask (via flasgger) |
| Highest throughput (>10K req/s) | Go net/http, Express.js, Litestar |
| Real-time streaming to clients | Express.js (SSE), FastAPI (StreamingResponse) |
| Full web app + AI backend | Django REST Framework |
| Async from the start | FastAPI, Litestar, Quart |
| Minimum latency budget | Go or Bun Elysia |
| Team familiar with Python | FastAPI (no context switch) |

**Recommendation for AI engineers:** Start with FastAPI. If you hit performance limits (rare), move the hot path to Go/Node while keeping the AI orchestration in FastAPI.
