# Infrastructure as Code (Conceptual)

IaC manages cloud resources using declarative configuration files instead of manual console clicks.

## Why IaC
- **Reproducible**: same config = same infrastructure every time
- **Version controlled**: track changes, review, rollback
- **Automated**: no manual errors, consistent across environments
- **Documentation**: config files serve as living documentation

## Tools
| Tool | Providers | Language |
|---|---|---|
| Terraform | AWS, GCP, Azure, many more | HCL |
| AWS CloudFormation | AWS only | YAML/JSON |
| Google Deployment Manager | GCP only | YAML |
| Azure ARM / Bicep | Azure only | JSON / DSL |
| Pulumi | AWS, GCP, Azure, many more | TypeScript, Python, Go |

## Example (Terraform)
```hcl
resource "aws_s3_bucket" "ml_data" {
  bucket = "my-project-ml-data"
  tags = {
    Project = "customer-churn"
  }
}

resource "aws_sagemaker_endpoint" "model" {
  name                 = "churn-model-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.model.name
}
```

## For AI Engineers
- Use IaC for reproducible ML environments
- Never manually create resources via console (except for quick experiments)
- Store IaC configs in the same repo as model code
- Use Terraform workspaces or directories for dev/staging/prod
