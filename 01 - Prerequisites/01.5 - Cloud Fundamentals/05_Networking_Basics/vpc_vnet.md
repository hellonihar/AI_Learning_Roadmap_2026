# Virtual Networks (VPC / VNet)

A VPC is your private network in the cloud — logically isolated from other tenants.

## Key Concepts
- **CIDR block**: IP range for your VPC (e.g., `10.0.0.0/16`)
- **Subnets**: divide VPC into segments (public / private)
- **Internet Gateway**: allows public internet access (for public subnets)
- **NAT Gateway**: allows private subnets to reach internet (for updates, API calls)
- **Route tables**: control traffic flow between subnets

## VPC Layout
```
Internet ─── Internet Gateway
                  │
          ┌───────┴───────┐
     Public Subnet    Private Subnet
     (web, LB)        (app, DB)
          │               │
          └─── NAT Gateway ──→ (for OS updates, API calls)
```

## For AI Engineers
- Place model inference endpoints in private subnets
- Place load balancers in public subnets
- Training clusters in private subnets with VPC interface endpoints for S3
- Use VPC Peering or Transit Gateway to connect multiple VPCs
