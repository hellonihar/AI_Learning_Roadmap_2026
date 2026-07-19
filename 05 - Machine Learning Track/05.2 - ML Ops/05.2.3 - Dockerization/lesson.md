# 05.2.3 Dockerization

## Docker Basics

### Images vs Containers

An **image** is a read-only template with instructions for creating a container. It includes the OS base layer, system dependencies, Python interpreter, libraries, application code, and model weights. Think of it as a **snapshot** or **class**.

A **container** is a runnable instance of an image — one or more running processes with their own filesystem, network, and process namespace. Think of it as an **object** instantiated from the image class. You can start, stop, move, and delete containers.

```bash
docker build -t ml-inference:latest .   # build an image
docker run -p 5000:5000 ml-inference     # run a container from the image
docker ps                                 # list running containers
docker stop <container_id>                # stop a container
```

### Why Docker for ML?

- **Environment consistency** — "Works on my machine" is eliminated. The same image runs identically on a laptop, CI server, and production GPU cluster.
- **Dependency isolation** — Each model (or pipeline step) can have its own Python version, CUDA toolkit, and libraries without conflicts.
- **Reproducibility** — The image digest (SHA256) is a cryptographic guarantee of the exact software stack.
- **Scalability** — Docker containers are the unit of deployment for Kubernetes, ECS, and other orchestration platforms.

## Dockerfile for ML

A well-structured Dockerfile for ML has four layers:

```dockerfile
# 1. Base image — pin a specific tag, never "latest"
FROM python:3.10-slim AS base

# 2. System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Python dependencies — pip install before copying code (layer caching)
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Application code and model weights
COPY src/ ./src/
COPY models/ ./models/

# 5. Entry point
EXPOSE 8080
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Layer Ordering Matters

Docker caches each layer. Put instructions that change infrequently **earlier** in the file:
- Base image (almost never changes).
- System dependencies (rare).
- Python dependencies (when `requirements.txt` changes).
- Application code (changes on every deploy).
- Model weights (changes when retrained).

This means a code-only change rebuilds only the last layer, taking seconds instead of minutes.

### Model Weights in Images

For small models (< 500 MB), bake weights into the image. For larger models:

- **Mount at runtime** via a volume: `docker run -v /path/to/models:/app/models ...`
- **Download on startup** from S3/GCS (requires credentials at runtime).
- **Use a sidecar container** with a shared volume (Kubernetes init container).

## docker-compose for Multi-Service

ML deployments often need multiple services: the inference API, a database, a caching layer (Redis), and monitoring. Docker Compose orchestrates these with a single YAML file.

```yaml
# docker-compose.yml
version: "3.9"
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
        condition: service_healthy

  redis:
    image: redis:alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ml_metadata
      POSTGRES_USER: mluser
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

Start everything: `docker-compose up -d`

## Best Practices

### Multi-Stage Builds

Multi-stage builds produce smaller, more secure images. Use a heavy "builder" stage for compilation and a minimal "runtime" stage for execution.

```dockerfile
# Stage 1: Build
FROM python:3.10-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim AS runtime
COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "src/app.py"]
```

The builder stage can contain gcc, CUDA toolkit, and build tools. The runtime stage contains only what's needed to run. The resulting image is often 300 MB vs 2 GB.

### .dockerignore

Prevent secrets, cache files, and large directories from being sent to the Docker daemon:

```dockerignore
__pycache__/
.git/
.vscode/
data/
notebooks/
*.pkl
.env
*.dvc
.dvc/
```

### Security

- **Never** bake secrets (API keys, DB passwords) into images. Use `--build-arg` for build-time secrets, environment variables or secrets managers for runtime secrets.
- **Avoid root** — add a non-root user in the Dockerfile.
- **Pin base image tags** — `python:3.10-slim` not `python:3-slim`.
- **Scan images** — `docker scout` or `trivy` to find CVEs.

```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

### GPU Support

Add `--gpus all` to `docker run` or in Compose:

```yaml
services:
  inference:
    image: ml-gpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Common Pitfalls

| Pitfall                        | Solution                                      |
| ------------------------------ | --------------------------------------------- |
| Image too large (> 5 GB)       | Multi-stage build, use slim base, prune cache |
| Secrets leaked in image layers | Use multi-stage, never COPY .env              |
| Slow rebuilds                  | Order layers by change frequency              |
| Permission errors at runtime   | Add non-root user, fix file ownership         |
| CUDA version mismatch          | Use NVIDIA CUDA base images, match host driver |

## Summary

Docker provides the environment standardization that ML systems need to go from a researcher's laptop to production. A well-crafted Dockerfile with proper layer ordering, multi-stage builds, and `.dockerignore` yields small, fast, secure images. Docker Compose ties together the inference service, databases, and caches for local development and small-scale production deployments.
