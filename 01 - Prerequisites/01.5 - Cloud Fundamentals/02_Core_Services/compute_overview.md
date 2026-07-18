# Compute Overview

## The Three Compute Layers

Cloud compute falls into three layers, each with different trade-offs:

### 1. Virtual Machines (IaaS)
Full control over the operating system, runtime, and hardware.

| Provider | Service |
|---|---|
| AWS | EC2 (Elastic Compute Cloud) |
| GCP | GCE (Google Compute Engine) |
| Azure | Virtual Machines |

**Significance**: You manage everything — OS, drivers, libraries, security patches. Best for workloads where you need complete control: GPU training with specific CUDA versions, legacy software, custom networking.

**Trade-off**: Highest operational overhead. You patch the OS, configure the network, handle failover.

### 2. Containers (PaaS-adjacent)
Package your application + dependencies into a portable image. The platform manages the OS and runtime; you manage the container.

| Provider | Service |
|---|---|
| AWS | ECS (Elastic Container Service), EKS (Kubernetes) |
| GCP | GKE (Google Kubernetes Engine), Cloud Run |
| Azure | AKS (Azure Kubernetes Service), Container Apps |

**Significance**: Reproducible across environments, fast startup (seconds), efficient resource utilization. The sweet spot for most AI workloads — containerized inference endpoints, training jobs, API servers.

**Trade-off**: You still manage the cluster (node scaling, upgrades) unless using serverless containers (Fargate, Cloud Run).

### 3. Serverless (Fully managed)
Upload code, platform handles everything else. No servers to provision or manage.

| Provider | Service |
|---|---|
| AWS | Lambda |
| GCP | Cloud Functions |
| Azure | Azure Functions |

**Significance**: Zero infrastructure management. Auto-scales from 0 to thousands of concurrent executions. Pay only for compute time used (per millisecond). Ideal for event-driven workloads and lightweight APIs.

**Trade-off**: Cold starts (1-2s after idle), 15-minute timeout limit, no GPU support, limited memory (10GB on Lambda).

---

## The Layer Trade-off Spectrum

```
Control ────────────────────────────────────────→ Convenience
High                                                  Low

[ VMs ] ──────→ [ Containers on VMs ] ──────→ [ Serverless ]
  │                      │                           │
  │ You manage:          │ You manage:                │ You manage:
  │  OS, runtime,        │  App code, config          │  App code only
  │  drivers, libs,      │  Container image           │
  │  security, scaling   │                            │
  │                      │ Platform manages:           │ Platform manages:
  │                      │  OS, runtime,              │  Everything
  │                      │  scaling, networking       │

Cost efficiency (for variable workloads):
  VMs: $$$$   (pay for provisioned capacity 24/7)
  Containers: $$-$$$ (pay for running containers, bin-packing)
  Serverless: $    (pay per invocation, down to zero)
```

---

## 5 Real-World Combinations

### 1. VM for Training → Containers for Inference

```
Training:     EC2 p5.48xlarge (bare VM, 8× H100, custom CUDA env)
              → You control GPU drivers, NCCL version, kernel params

Artifact:     Model weights uploaded to S3

Inference:    ECS (Docker containers on EC2)
              → Containerized model server, auto-scaled by ALB

Why this combo: Training needs raw GPU access and full stack control.
Inference benefits from container portability and auto-scaling.
Used by: Most ML teams deploying custom models.
```

### 2. All-Serverless Event Pipeline

```
Trigger:      S3 upload event
Process:      Lambda (clean data, validate schema)
Store:        DynamoDB (metadata), S3 (processed data)
Serve:        API Gateway + Lambda (lightweight inference)

Why this combo: Zero infrastructure. Scales to zero when idle.
Perfect for sporadic workloads (user uploads, scheduled jobs).
Limitation: No GPU inference, cold start latency.
Used by: Data preprocessing pipelines, lightweight classification.
```

### 3. Serverless Containers (Fargate / Cloud Run)

```
Service:      FastAPI model server
Compute:      AWS Fargate (no cluster management)
Scaling:      Target tracking based on request count
Networking:   ALB → Fargate tasks (multi-AZ)

Why this combo: No EC2 instances to manage, no cluster upgrades.
Containers give you GPU support (via Fargate GPU on some providers),
predictable startup, and per-task billing.
Used by: Teams that want container benefits without ops overhead.
```

### 4. Hybrid: Lambda + ECS for Long Tasks

```
Frontend:     API Gateway + Lambda (authentication, routing, caching)
Async task:   Lambda submits job to SQS queue
Worker:       ECS on EC2 (GPU instance processes the job)
Result:       Written to S3, Lambda polls status

Why this combo: Serverless handles bursty short-lived requests cheaply.
Containers handle long-running GPU inference that exceeds Lambda limits.
Used by: AI apps with a mix of quick lookups and heavy computation.
```

### 5. Kubernetes Everywhere (EKS / GKE / AKS)

```
All services:  Deployed on a single Kubernetes cluster
  - Model training:    Job (batch, spot instances)
  - Model serving:     Deployment + Service (auto-scaled by CPU/latency)
  - Feature pipeline:  CronJob (runs hourly)
  - API gateway:       Ingress controller (NGINX / Traefik)
  - Monitoring:        Prometheus + Grafana in-cluster

Why this combo: Unified control plane. Same deployment model for
everything. Portability across clouds. Advanced scheduling (GPU
node pools, taints/tolerations).
Used by: Organizations with mature Kubernetes teams running
multiple ML workloads on shared cluster.
```

---

## Choosing the Right Combination

| Scenario | Recommended | Why |
|---|---|---|
| GPU training, custom env | VM (EC2/GCE) | Full control of drivers, CUDA, networking |
| Standard inference API | Containers (ECS/GKE) | Portable, auto-scale, good GPU support |
| Simple webhook / data proc | Serverless (Lambda) | Zero ops, scales to zero, cheap |
| Mixed workloads, team of 5+ | Kubernetes | Unified infra, portability, rich scheduling |
| Quick MVP, no ops team | Serverless containers (Fargate/Cloud Run) | Container benefits without cluster management |
