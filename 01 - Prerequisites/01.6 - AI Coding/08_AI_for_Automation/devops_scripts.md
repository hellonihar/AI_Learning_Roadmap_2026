# DevOps-Related Scripts

Generate Dockerfiles, Terraform configs, deployment scripts, and monitoring tools.

## Prompt

```
"Write a Dockerfile for a FastAPI model serving application that:
- Uses python:3.12-slim as base
- Installs system dependencies for PyTorch
- Creates a non-root user for security
- Copies only the requirements first (layer caching)
- Installs dependencies
- Copies the rest of the application
- Exposes port 8000
- Uses uvicorn with 8 workers
- Includes a health check"
```

## Example

```dockerfile
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
RUN groupadd -r app && useradd -r -g app -d /app app
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --chown=app:app . .
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "8"]
```

## DevOps Prompts

- "Write a Terraform module for deploying SageMaker endpoint with auto-scaling"
- "Generate a docker-compose.yml for local development with Redis, Postgres, and the app"
- "Create a startup script that initializes a GPU instance: install drivers, mount volumes, pull Docker image"
- "Write a script to collect GPU metrics (utilization, memory, temperature) and send to CloudWatch"
- "Generate a Kubernetes deployment YAML with resource limits, probes, and horizontal pod autoscaling"
