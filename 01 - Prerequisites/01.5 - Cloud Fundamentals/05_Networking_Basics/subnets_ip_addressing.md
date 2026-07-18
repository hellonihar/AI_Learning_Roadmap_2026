# Subnets & IP Addressing

## Subnets
- A subnet is a range of IP addresses within a VPC
- Each subnet maps to a single Availability Zone
- Cannot span AZs

## Public vs Private Subnets
| Feature | Public Subnet | Private Subnet |
|---|---|---|
| Internet access | Yes (via IGW) | No (unless via NAT) |
| Route table entry | 0.0.0.0/0 → IGW | 0.0.0.0/0 → NAT |
| Use cases | Load balancers, bastion hosts | App servers, databases |

## IP Addressing
- **Private IP**: internal only, from VPC CIDR range
- **Public IP**: internet-routable, assigned by AWS/GCP
- **Elastic IP**: static public IP you can reassign
- **CIDR notation**: `10.0.0.0/16` = 65,536 IPs, `10.0.1.0/24` = 256 IPs

## For AI Engineers
- Use private subnets for all compute workloads
- Use a bastion host (jump box) in a public subnet to SSH into private instances
- Or use AWS Systems Manager Session Manager for secure access without public IPs
