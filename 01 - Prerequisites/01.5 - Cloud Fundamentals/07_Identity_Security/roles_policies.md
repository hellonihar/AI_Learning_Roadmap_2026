# Roles & Policies

## When to Use Roles
- EC2 instance needs to read from S3 → attach an IAM role to the instance
- Lambda function needs to write to DynamoDB → assign execution role
- Cross-account access → role with trust policy from other account

## AWS Managed Policies (common ones)
| Policy | Grants |
|---|---|
| AmazonS3FullAccess | Full S3 read/write |
| AmazonSageMakerFullAccess | Full SageMaker access |
| AWSLambdaBasicExecutionRole | Write CloudWatch logs |
| AmazonECS_FullAccess | Full ECS control |

## Policy Types
| Type | Description |
|---|---|
| Identity-based | Attached to user/group/role |
| Resource-based | Attached to resource (S3 bucket policy) |
| Permissions boundary | Max permissions a role can have |
| Service control policy (SCP) | Org-level boundary (AWS Organizations) |

## For AI Engineers
- Create custom policies scoped to specific S3 paths, specific SageMaker actions
- Use `*` sparingly — specify exact ARNs and actions
- Use `Condition` to restrict by IP, VPC, or time of day
- Test policies with IAM Access Analyzer (AWS) or Policy Troubleshooter (GCP)
