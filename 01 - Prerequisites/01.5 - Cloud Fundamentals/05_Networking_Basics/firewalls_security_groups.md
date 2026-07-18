# Firewalls & Security Groups

## Security Groups (stateful firewall)
- Act as virtual firewall for EC2 / ECS / RDS instances
- **Stateful**: return traffic is automatically allowed
- Default: all inbound blocked, all outbound allowed

```yaml
# Example: Web server security group
Inbound rules:
  - HTTP (80)   from 0.0.0.0/0
  - HTTPS (443) from 0.0.0.0/0
  - SSH (22)    from 10.0.0.0/16 (bastion only)
Outbound rules:
  - All traffic to 0.0.0.0/0
```

## Network ACLs (stateless firewall)
- Apply at subnet level
- **Stateless**: return traffic must be explicitly allowed
- Rules evaluated by rule number (lowest first)

## For AI Engineers
- Restrict SSH access to bastion hosts only
- Allow inference endpoints on port 443 from the load balancer security group
- Allow training cluster nodes to communicate on a private range (e.g., port 22, 443, 8000-9000)
- Never open port 22 to 0.0.0.0/0
