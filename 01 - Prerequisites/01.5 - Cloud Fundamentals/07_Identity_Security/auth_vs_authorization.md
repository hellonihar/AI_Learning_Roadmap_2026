# Authentication vs Authorization

## Authentication (AuthN)
- **Who are you?** Verification of identity
- Methods: password, MFA, SSO, OIDC, IAM keys
- **Sign-in** process

## Authorization (AuthZ)
- **What are you allowed to do?** Granting permissions
- Based on policies assigned to authenticated identity
- **Access control** after sign-in

## Cloud Examples
```
Authentication:                Authorization:
  IAM user + password  ──────→  IAM policies attached
  Access Key + Secret  ──────→  Assigned IAM role permissions
  OIDC (GitHub Actions) ──────→  Assume role via trust policy
  Service Account Key  ──────→  IAM roles on GCP
```

## For AI Engineers
- Service accounts / IAM roles are the correct identity for ML pipelines — not human users
- GitHub Actions assumes an IAM role via OIDC to deploy models
- Training jobs assume a service account with S3 read access
- Inference endpoints assume a role with model artifact read access
