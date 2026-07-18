# Regions & Availability Zones

## Regions
- Geographic area with 2+ availability zones
- Choose regions based on:
  - **Latency**: deploy close to users
  - **Compliance**: data must stay in specific countries
  - **Service availability**: not all services in all regions
  - **Cost**: prices vary by region

## Availability Zones (AZs)
- Isolated data centers within a region (separate power, cooling, network)
- Connected via low-latency fiber
- **Use multiple AZs** for high availability

## For AI Engineers
- Train in cheap regions (us-east-1), deploy inference close to users
- Use multi-region storage for datasets that must be globally accessible
- Design for AZ failure: don't put all GPUs in one AZ
- Some GPU instance types are region-restricted — check availability first
