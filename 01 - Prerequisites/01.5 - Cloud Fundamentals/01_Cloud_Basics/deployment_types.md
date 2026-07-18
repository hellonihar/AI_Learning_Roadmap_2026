# Deployment Types

## Public Cloud
- Provider owns all infrastructure
- Multi-tenant (your data is isolated)
- **Pros**: no capex, elastic, pay-as-you-go
- **Examples**: AWS, GCP, Azure

## Private Cloud
- Dedicated infrastructure for one organization
- On-premises or hosted by provider
- **Pros**: full control, compliance, data residency
- **Cons**: higher cost, you manage it

## Hybrid Cloud
- Connect public cloud + on-premises via VPN or Direct Connect
- **Use case**: burst to cloud for GPU training, keep sensitive data on-prem

## Multi-Cloud
- Use 2+ public cloud providers
- **Use case**: avoid lock-in, use best service per workload
