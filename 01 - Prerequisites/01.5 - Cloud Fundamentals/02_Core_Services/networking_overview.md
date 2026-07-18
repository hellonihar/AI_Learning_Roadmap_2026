# Networking Overview

## Virtual Network (VPC / VNet)
- Isolated network within the cloud
- You control IP range (CIDR), subnets, route tables
- **Default**: all traffic blocked inbound, allowed outbound

## Subnets
- Divide VPC into public (internet-facing) and private (internal)
- Public subnets have route to internet gateway
- Private subnets accessed via NAT gateway or VPN

## DNS (Route53, Cloud DNS)
- Translates domain names to IP addresses
- Route traffic to load balancers, CDN, or specific IPs
- Supports latency-based, geo-based, weighted routing

## CDN (CloudFront, Cloud CDN)
- Caches content at edge locations near users
- Reduces latency for static assets (images, model inference UI)
- Can also cache API responses

## For AI Engineers
- Model inference endpoints sit behind load balancers in private subnets
- Access from internet goes through API Gateway → LB → service
- Training clusters use internal networking for high-throughput data transfer
