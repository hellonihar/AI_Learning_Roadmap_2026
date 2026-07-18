# Software Essentials Quick Reference

## Git
```bash
git init / clone <url>
git add -A / commit -m "msg" / push / pull
git checkout -b feature-branch
git merge feature-branch
git stash / git stash pop
git log --oneline --graph
git revert <hash>
git cherry-pick <hash>
```

## Docker
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

```yaml
# docker-compose.yml
services:
  app:
    build: .
    ports: ["8000:8000"]
    volumes: [".:/app"]
```

## API Calls
```python
import httpx
resp = httpx.get("https://api.example.com/data",
    headers={"Authorization": "Bearer key"})
resp.raise_for_status()
data = resp.json()
```

## OpenAI API
```python
from openai import OpenAI
client = OpenAI()
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "hi"}],
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

## FastAPI
```python
@app.post("/chat")
def chat(req: ChatRequest) -> ChatResponse:
    reply = openai_client.chat(...)
    return ChatResponse(reply=reply)
```

## VS Code Extensions
- Python, Pylance, Ruff, GitLens, Docker, Jupyter, GitHub Copilot
