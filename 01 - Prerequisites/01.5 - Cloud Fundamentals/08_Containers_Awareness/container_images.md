# Container Images

## Image Layers
Each Dockerfile instruction creates a layer. Layers are cached and reused.

```dockerfile
FROM python:3.12-slim          # base layer (cached)
WORKDIR /app                    # layer (cached)
COPY requirements.txt .         # layer (cached unless reqs change)
RUN pip install -r reqs.txt     # layer (cached unless reqs change)
COPY . .                        # layer (changes often)
```

## Best Practices
- Use specific tags (`python:3.12-slim`), not `latest`
- Minimize layers: combine RUN commands with `&&`
- Remove package manager cache: `apt-get clean && rm -rf /var/lib/apt/lists/*`
- Use `.dockerignore` to avoid copying unnecessary files
- Scan images for vulnerabilities (`docker scout`, `trivy`)

## Tagging Strategy
```bash
docker build -t my-model:v1.2.3 .
docker build -t my-model:latest .
docker tag my-model:v1.2.3 registry.example.com/my-model:v1.2.3
docker push registry.example.com/my-model:v1.2.3
```
