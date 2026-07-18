# Cost Estimation & Calculators

## Official Pricing Calculators
| Provider | Tool |
|---|---|
| AWS | [AWS Pricing Calculator](https://calculator.aws/) |
| GCP | [Google Cloud Pricing Calculator](https://cloud.google.com/products/calculator) |
| Azure | [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/) |

## What to Estimate
| Item | Example Cost (us-east-1, 2026) |
|---|---|
| p5.48xlarge (8x H100) on-demand | ~$100/hr |
| p5.48xlarge spot | ~$25-30/hr |
| g5.xlarge (1x A10G) on-demand | ~$1/hr |
| S3 Standard (100 TB) | ~$2,300/month |
| Data transfer out (10 TB) | ~$900/month |

## Cost Optimization Checklist
- [ ] Use spot instances for training
- [ ] Delete idle instances (don't leave GPU instances running overnight)
- [ ] Use auto-scaling to reduce to zero when not needed
- [ ] Set S3 lifecycle policies (standard → IA → Glacier)
- [ ] Use reserved capacity for steady-state workloads
- [ ] Monitor cross-region data transfer costs
