# Environment Separation (Dev, Staging, Prod)

## Why Separate Environments
- **Dev**: experiment freely, break things, no impact on users
- **Staging**: validate changes in a production-like environment
- **Prod**: serve real users with strict access controls

## Separation Strategies
| Aspect | Dev | Staging | Prod |
|---|---|---|---|
| Cost | Spot, small instances | Reserved, scaled-down | Reserved, production scale |
| Access | Team | Team + QA | Strict IAM, MFA required |
| Data | Synthetic / sample | Anonymized production | Real user data |
| GPU | 1× T4 | 1× A10G | Multiple H100 |
| Deploy | Every commit | On merge to main | Tagged releases |
| Alerts | None | Warnings | PagerDuty |

## How to Implement
- Separate AWS accounts (recommended) or separate VPCs
- Terraform workspaces / directories per environment
- CI/CD pipeline promotes: dev → staging → prod (with approvals)
- Use naming convention: `my-project-dev`, `my-project-staging`, `my-project-prod`
