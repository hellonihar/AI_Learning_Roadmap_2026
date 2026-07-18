# Docker for Development

## Dockerfile for AI App
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Docker Compose (Dev Environment)
```yaml
# docker-compose.yml
version: "3.9"
services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app          # live reload
      - ~/.cache/huggingface:/root/.cache/huggingface  # cache models
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    command: uvicorn main:app --reload --host 0.0.0.0

  db:
    image: postgres:16
    environment:
      POSTGRES_DB: aiexperiments
      POSTGRES_PASSWORD: devpassword
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

## Dev Containers (VS Code)
```json
// .devcontainer/devcontainer.json
{
    "name": "AI Dev Environment",
    "build": { "dockerfile": "Dockerfile" },
    "settings": { "python.defaultInterpreterPath": "/usr/local/bin/python" },
    "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "charliermarsh.ruff",
        "github.copilot"
    ],
    "postCreateCommand": "pip install -r requirements.txt"
}
```
