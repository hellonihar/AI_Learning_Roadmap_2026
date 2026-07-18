# Pay-as-you-go Pricing

## What You Pay For
| Resource | Billed By |
|---|---|
| Compute (EC2, ECS) | Per second/minute, based on instance type |
| Storage (S3, EBS) | Per GB per month |
| Data transfer | Per GB (inbound is free, outbound is charged) |
| API calls | Per 1,000 requests (Lambda, S3, API Gateway) |
| GPUs | Per hour (significantly more expensive than CPU) |

## Pricing Models
| Model | Savings | Commitment |
|---|---|---|
| On-Demand | 0% | None |
| Reserved (1yr) | 30-40% | 1 year |
| Reserved (3yr) | 50-60% | 3 years |
| Spot | 60-90% | None (can be interrupted) |
| Savings Plan | 30-60% | $/hr commitment for 1 or 3 years |

## For AI Engineers
- Training: use spot GPU instances (save 60-70%), checkpoint frequently
- EBS: snapshot and delete volumes after training completes
- S3: use lifecycle policies to move old data to Glacier (90% cheaper)
- Monitor data egress costs (cross-region, internet) — can be significant
