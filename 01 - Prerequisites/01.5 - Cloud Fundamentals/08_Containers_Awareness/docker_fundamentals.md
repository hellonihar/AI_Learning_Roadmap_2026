# Docker Fundamentals

## Key Concepts
- **Image**: read-only template (OS + app + dependencies)
- **Container**: running instance of an image
- **Dockerfile**: recipe to build an image
- **Volume**: persistent data storage for containers
- **Network**: how containers communicate

## Dockerfile Example
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

## Common Commands
```bash
docker build -t my-image:latest .
docker run -p 8000:8000 my-image:latest
docker ps                    # running containers
docker images                # local images
docker exec -it <id> bash    # shell into container
docker logs <id>             # view logs
docker compose up            # multi-container app
```

## For AI Engineers
- Containerize models for reproducible deployments
- Use `.dockerignore` to exclude `__pycache__/`, `.venv/`, large datasets
- Use multi-stage builds to keep images small
- Tag images with git commit hash for traceability
