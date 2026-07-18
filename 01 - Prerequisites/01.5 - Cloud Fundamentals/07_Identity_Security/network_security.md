# Network Security Basics

## Defense in Depth
```
Internet → WAF → ALB → Security Group → App → IAM → DB
```

## Security Layers
| Layer | Tool | Purpose |
|---|---|---|
| Edge | AWS WAF, CloudFront | Block common web exploits, DDoS protection |
| Network | Security Groups | Instance-level firewall (stateful) |
| Subnet | NACLs | Subnet-level firewall (stateless) |
| Application | IAM, API Gateway | Auth, rate limiting, request validation |
| Data | KMS, SSL/TLS | Encryption at rest and in transit |

## Common Threats & Mitigations
| Threat | Mitigation |
|---|---|
| Port scanning | Security groups block all inbound by default |
| DDoS | Shield (AWS) / Cloud Armor (GCP) / DDoS Protection (Azure) |
| Data exfiltration | S3 bucket policies, VPC endpoints, least IAM |
| Credential leak | Secrets manager, IAM roles, short-lived tokens |

## For AI Engineers
- Model endpoints should only be accessible via internal network or API Gateway
- Training data in S3 should be encrypted and accessed via VPC endpoints (not over internet)
- Never open SSH (22) to the world — use Session Manager or bastion host
- Use WAF to protect inference APIs from prompt injection attempts
