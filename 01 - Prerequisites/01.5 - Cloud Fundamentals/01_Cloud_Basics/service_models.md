# Cloud Service Models (IaaS, PaaS, SaaS)

```
┌─────────────────────────────────────┐
│          SaaS (ready-to-use)        │  Gmail, Slack, GitHub
├─────────────────────────────────────┤
│    PaaS (managed runtime)           │  Heroku, Cloud Run, RDS
├─────────────────────────────────────┤
│    IaaS (virtualized infra)         │  EC2, GCE, S3
├─────────────────────────────────────┤
│           On-Premises               │  you manage everything
└─────────────────────────────────────┘
```

## IaaS — Infrastructure as a Service
- VMs, storage, networking, load balancers
- You manage OS, runtime, app, data
- **For AI**: EC2 GPU instances, persistent block storage

## PaaS — Platform as a Service
- Managed runtime, database, message queues
- You just deploy code
- **For AI**: SageMaker, Vertex AI, Cloud Run

## SaaS — Software as a Service
- Fully managed application
- You just use the product
- **For AI**: ChatGPT, GitHub Copilot
