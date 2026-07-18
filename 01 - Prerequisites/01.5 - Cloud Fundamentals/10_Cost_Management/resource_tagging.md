# Resource Tagging

Tags are key-value pairs attached to cloud resources for organization, cost allocation, and automation.

## Common Tags
```yaml
Project: "customer-churn"
Environment: "production"
Owner: "ml-team"
CostCenter: "12345"
DataClassification: "confidential"
CreatedBy: "terraform"
```

## Benefits
- **Cost allocation**: filter billing reports by project or team
- **Automation**: stop resources tagged `Environment: dev` outside business hours
- **Visibility**: identify orphaned resources
- **Compliance**: enforce tags with policies

## Tagging Strategy
```
Required tags:
  Project, Environment, Owner, CostCenter

Applied automatically via:
  - Infrastructure as Code (Terraform)
  - Org-level tag policies (AWS Organizations)
  - Default tags in deployment pipeline
```

## For AI Engineers
- Tag GPU instances with `Project: {experiment-name}` for cost tracking
- Tag S3 buckets with `DataClassification` to enforce security controls
- Tag SageMaker endpoints with `ModelName` + `Version`
- Use tags to auto-stop dev resources after hours
