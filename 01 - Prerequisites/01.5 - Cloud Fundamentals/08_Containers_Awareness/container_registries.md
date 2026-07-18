# Container Registries

A registry stores and distributes container images.

## Major Registries
| Registry | Example URL | Best For |
|---|---|---|
| Docker Hub | `nginx:latest` | Public images, base images |
| Amazon ECR | `123.dkr.ecr.region.amazonaws.com/my-image` | AWS deployments |
| Google Artifact Registry | `us-central1-docker.pkg.dev/my-project/my-repo` | GCP deployments |
| Azure Container Registry | `myregistry.azurecr.io/my-image` | Azure deployments |
| GitHub Container Registry | `ghcr.io/my-org/my-image` | CI/CD integration |

## Commands
```bash
aws ecr get-login-password | docker login --username AWS --password-stdin <registry>
docker tag my-image:latest <registry>/my-image:latest
docker push <registry>/my-image:latest
docker pull <registry>/my-image:latest
```
