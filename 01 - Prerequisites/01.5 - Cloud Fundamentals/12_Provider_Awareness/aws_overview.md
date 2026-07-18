# AWS Overview

## Key Services for AI
| Category | Service |
|---|---|
| Compute | EC2 (GPU: p5, p4d, g5), ECS / EKS, Lambda |
| Storage | S3, EBS, EFS, FSx for Lustre |
| ML Platform | SageMaker (training, endpoints, notebooks) |
| AI Services | Bedrock (foundation models), Rekognition, Comprehend, Polly |
| Databases | RDS, DynamoDB, ElastiCache |
| Networking | VPC, ALB, Route53, CloudFront |
| Security | IAM, KMS, Secrets Manager, WAF |
| CI/CD | CodePipeline, CodeBuild, CodeDeploy |

## Strengths
- Largest market share, most services, most mature ecosystem
- SageMaker is the most comprehensive managed ML platform
- Extensive GPU instance options (p5 H100, p4d A100, g5 A10G)
- Bedrock gives access to foundation models without managing infra

## Considerations
- Can be overwhelming (200+ services)
- Networking can be complex (VPCs, transit gateway, peering)
- Cost management requires attention (easy to overspend)
